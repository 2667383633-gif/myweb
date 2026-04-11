import numpy as np
import cv2
import os
import argparse
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt  # kept for compatibility with original script
from segment_anything import sam_model_registry, SamPredictor
import SimpleITK as sitk

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks"
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--mask-input",
    type=str,
    required=True,
    help="Path to either a single crf mask image or folder of crf mask images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument(
    "--prompts",
    type=str,
    required=True,
    help="The type of prompts to use, in ['points', 'boxes', 'both']",
)

parser.add_argument(
    "--num-points",
    type=int,
    required=False,
    default=10,
    help="Number of points when using point prompts, default is 8",
)

parser.add_argument(
    "--negative",
    action="store_true",
    help="Whether to sample points in the background. Default is False.",
)

parser.add_argument(
    "--neg-num-points",
    type=int,
    required=False,
    default=10,
    help="Number of negative points when using negative mode, default is 20",
)

parser.add_argument(
    "--pos-margin",
    type=float,
    required=False,
    default=10,
    help=(
        "controls the sampling margin for the positive point prompts, default is 2, "
        "for large structures use above 15, but for smaller objects use 2-5"
    ),
)

parser.add_argument(
    "--neg-margin",
    type=float,
    required=False,
    default=5,
    help="controls the sampling margin for the negative point prompts, default is 5",
)

parser.add_argument(
    "--multimask",
    action="store_true",
    help="Whether to output multimasks in SAM. Default is False.",
)

parser.add_argument(
    "--multicontour",
    action="store_true",
    help="Whether to output multiple bounding boxes for each contour. Default is False.",
)

parser.add_argument(
    "--voting",
    type=str,
    default="AVERAGE",
    help="['MRM','STAPLE','AVERAGE']",
)

parser.add_argument(
    "--plot",
    action="store_true",
    help="Whether to plot the points and boxes in the contours. Default is False.",
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="The device to run generation on.",
)


# ==============================
# Utilities for Multi‑Prompt Consistency
# ==============================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def calculate_iou(maskA: np.ndarray, maskB: np.ndarray, eps: float = 1e-6) -> float:
    """Compute IoU for two boolean/binary masks of the same size with epsilon guard."""
    a = (maskA > 0).astype(np.uint8)
    b = (maskB > 0).astype(np.uint8)
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    union = np.logical_or(a, b).sum(dtype=np.int64)
    return float(inter) / float(max(union, 0) + eps)


def write_mask_to_folder(mask, t_mask, path: str, num_contours: int) -> None:
    """Save binary mask and keep only the top-K connected components (K=num_contours)."""
    file = os.path.basename(t_mask)
    filename = f"{file}"

    mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]
    mask = mask.astype(np.uint8) * 255

    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(mask)
    sizes = stats[:, cv2.CC_STAT_AREA]
    sorted_sizes = sorted(sizes[1:], reverse=True)  # ignore background

    top_k_sizes = sorted_sizes[: max(0, num_contours)] if num_contours > 0 else []

    im_result = np.zeros_like(im_with_separated_blobs)
    if len(top_k_sizes) > 0:
        for index_blob in range(1, nb_blobs):
            if sizes[index_blob] in top_k_sizes:
                im_result[im_with_separated_blobs == index_blob] = 255

    os.makedirs(path, exist_ok=True)
    cv2.imwrite(os.path.join(path, filename), im_result)


def scoremap2bbox(scoremap: np.ndarray, multi_contour_eval: bool = False):
    """Return (estimated_boxes, contours, num_contours) from a binary map in [0,1] or [0,255]."""
    scoremap = (scoremap > 0).astype(np.uint8)
    h, w = scoremap.shape

    contours, _ = cv2.findContours(scoremap, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)

    if num_contours == 0:
        return np.asarray([[0, 0, w - 1, h - 1]]), [], 0

    if not multi_contour_eval:
        contours = [np.concatenate(contours)]

    estimated_boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + bw - 1, y + bh - 1
        x1 = min(x1, w - 1)
        y1 = min(y1, h - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes, dtype=np.int32), contours, num_contours


def largest_cc_bbox(mask_bin: np.ndarray, pad_ratio: float = 0.05) -> np.ndarray:
    """Get bbox [x0,y0,x1,y1] of largest CC in mask_bin, padded by pad_ratio of box size."""
    h, w = mask_bin.shape[:2]
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin.astype(np.uint8))
    if nb <= 1:
        return np.array([0, 0, w - 1, h - 1], dtype=np.int32)

    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    idx = int(1 + np.argmax(areas))
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    bw = int(stats[idx, cv2.CC_STAT_WIDTH])
    bh = int(stats[idx, cv2.CC_STAT_HEIGHT])

    x0, y0, x1, y1 = x, y, x + bw - 1, y + bh - 1
    dx = int(round(pad_ratio * bw))
    dy = int(round(pad_ratio * bh))
    x0 = max(0, x0 - dx)
    y0 = max(0, y0 - dy)
    x1 = min(w - 1, x1 + dx)
    y1 = min(h - 1, y1 + dy)
    return np.array([x0, y0, x1, y1], dtype=np.int32)


def positive_point_by_dist_transform(mask_bin: np.ndarray) -> np.ndarray:
    """Return a foreground point guaranteed inside the mask using distance transform peak."""
    dt = cv2.distanceTransform((mask_bin > 0).astype(np.uint8), cv2.DIST_L2, 5)
    if dt.max() <= 0:
        ys, xs = np.where(mask_bin > 0)
        if len(xs) == 0:
            return np.array([0, 0], dtype=np.int32)
        k = np.random.randint(0, len(xs))
        return np.array([int(xs[k]), int(ys[k])], dtype=np.int32)
    y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return np.array([int(x), int(y)], dtype=np.int32)


def sample_negative_points(mask_bin: np.ndarray, num_points: int = 2, kernel: int = 15) -> np.ndarray:
    """Sample negative (background) points from a dilated ring around the mask.
    Fallback to random background if the ring is empty.
    """
    if kernel % 2 == 0:
        kernel += 1
    st = np.ones((kernel, kernel), dtype=np.uint8)
    dil = cv2.dilate((mask_bin > 0).astype(np.uint8), st)
    ring = ((dil > 0) & (mask_bin == 0))

    ys, xs = np.where(ring)
    pts = []
    if len(xs) >= 1:
        idxs = np.random.choice(len(xs), size=min(num_points, len(xs)), replace=False)
        for i in idxs:
            pts.append([int(xs[i]), int(ys[i])])

    while len(pts) < num_points:
        bys, bxs = np.where(mask_bin == 0)
        if len(bxs) == 0:
            pts.append([0, 0])
        else:
            j = np.random.randint(0, len(bxs))
            pts.append([int(bxs[j]), int(bys[j])])

    return np.asarray(pts[:num_points], dtype=np.int32)


def main(args: argparse.Namespace) -> None:
    print("Segmenting images using SAM...")

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)

    predictor = SamPredictor(sam)
    set_seed(42)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    if not os.path.isdir(args.mask_input):
        targets_mask = [args.mask_input]
    else:
        targets_mask = [
            f for f in os.listdir(args.mask_input) if not os.path.isdir(os.path.join(args.mask_input, f))
        ]
        targets_mask = [os.path.join(args.mask_input, f) for f in targets_mask]

    os.makedirs(args.output, exist_ok=True)

    for t, t_mask in tqdm(zip(targets, targets_mask), total=len(targets)):
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(t_mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Could not load mask '{t_mask}', skipping...")
            continue

        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # If coarse mask empty, save black and continue
        if mask.max() == 0:
            write_mask_to_folder(np.zeros_like(mask, dtype=np.uint8), t_mask, args.output, num_contours=0)
            continue

        # Determine number of contours (kept for output policy)
        _, _, num_contours = scoremap2bbox((mask > 0).astype(np.uint8), multi_contour_eval=True)

        # Build prompts from coarse mask
        mask_bin = (mask > 0).astype(np.uint8)
        box = largest_cc_bbox(mask_bin, pad_ratio=0.05).astype(np.int32)
        pos_pt = positive_point_by_dist_transform(mask_bin)
        neg_pts = sample_negative_points(mask_bin, num_points=2, kernel=15)

        # Run 1: Box only
        predictor.set_image(image)
        masks1, scores1, _ = predictor.predict(box=box, multimask_output=True)
        m1 = masks1[np.argmax(scores1)].astype(np.uint8)

        # Run 2: Box + positive point
        predictor.set_image(image)
        masks2, scores2, _ = predictor.predict(
            point_coords=np.array([pos_pt], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            box=box,
            multimask_output=True,
        )
        m2 = masks2[np.argmax(scores2)].astype(np.uint8)

        # Run 3: Box + positive + two negative points
        predictor.set_image(image)
        mp = np.vstack([pos_pt[None, :], neg_pts]).astype(np.float32)
        ml = np.array([1, 0, 0], dtype=np.int32)
        masks3, scores3, _ = predictor.predict(
            point_coords=mp,
            point_labels=ml,
            box=box,
            multimask_output=True,
        )
        m3 = masks3[np.argmax(scores3)].astype(np.uint8)

        # Consistency evaluation
        iou_12 = calculate_iou(m1, m2)
        iou_13 = calculate_iou(m1, m3)
        iou_23 = calculate_iou(m2, m3)
        scores = [iou_12 + iou_13, iou_12 + iou_23, iou_13 + iou_23]
        final_mask = [m1, m2, m3][int(np.argmax(scores))].astype(float)

        # Save the final mask (preserving prior top-k connected component filtering)
        write_mask_to_folder(final_mask, t_mask, args.output, num_contours)

    print("SAM Segmentation Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
