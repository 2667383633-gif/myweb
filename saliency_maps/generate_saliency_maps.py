import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import os
import cv2
import json
import math
import time
import torch
import random
import argparse
import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from scripts.plot import visualize_vandt_heatmap
from scripts.methods import vision_heatmap_iba

try:
    from prompt_router import PromptRouter
except Exception:
    PromptRouter = None


# =========================================================
# Utils
# =========================================================
def get_text_features(model, tokenizer, text, device="cuda"):
    if isinstance(text, str):
        text = [text]

    toks = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    toks = {k: v.to(device) for k, v in toks.items()}

    with torch.no_grad():
        if hasattr(model, "get_text_features"):
            text_features = model.get_text_features(**toks)
        else:
            outputs = model.text_model(**toks)
            text_features = outputs.pooler_output

    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-12)
    return text_features


def get_img_features(
    model,
    processor,
    img_bgr,
    device="cuda",
    return_img_tensor=False,
    return_patch_tokens=False,
):
    # OpenCV BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    proc = processor(images=img_rgb, return_tensors="pt")
    pixel_values = proc["pixel_values"].to(device)

    with torch.no_grad():
        vision_outputs = None
        if hasattr(model, "vision_model"):
            vision_outputs = model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True
            )

        if hasattr(model, "get_image_features"):
            image_features = model.get_image_features(pixel_values=pixel_values)
        elif vision_outputs is not None:
            if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
                image_features = vision_outputs.pooler_output
            else:
                image_features = vision_outputs.last_hidden_state[:, 0, :]
        else:
            raise AttributeError("Model does not expose get_image_features or vision_model.")

    image_features = F.normalize(image_features, dim=-1)

    outputs = [image_features]

    if return_img_tensor:
        outputs.append(pixel_values)

    if return_patch_tokens:
        if vision_outputs is None:
            with torch.no_grad():
                vision_outputs = model.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True
                )

        if hasattr(vision_outputs, "last_hidden_state"):
            tokens = vision_outputs.last_hidden_state
        else:
            tokens = vision_outputs[0]

        if tokens.dim() != 3 or tokens.shape[1] <= 1:
            raise ValueError("Unable to extract patch tokens from vision encoder output.")

        # remove CLS token
        patch_tokens = tokens[:, 1:, :]
        patch_tokens = F.normalize(patch_tokens, dim=-1)
        outputs.append(patch_tokens)

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_images(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()
    return files


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return mask


def normalize_map(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def resize_saliency(vmap: np.ndarray, w: int, h: int):
    vmap = np.asarray(vmap, dtype=np.float32)
    return cv2.resize(vmap, (w, h), interpolation=cv2.INTER_LINEAR)


def save_saliency(path: str, vmap: np.ndarray):
    vmap = normalize_map(vmap)
    out = np.clip(vmap * 255.0, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(path, out)
    if not ok:
        raise IOError(f"Failed to save saliency map to {path}")


def load_saved_saliency(path: str):
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    return arr.astype(np.float32) / 255.0


def mask_to_bool(mask):
    return np.asarray(mask) > 0


def calculate_dice_coefficient(gt_mask, pred_mask, eps=1e-6):
    gt = mask_to_bool(gt_mask)
    pr = mask_to_bool(pred_mask)
    inter = np.logical_and(gt, pr).sum()
    denom = gt.sum() + pr.sum()
    if denom == 0:
        return 1.0
    return (2.0 * inter) / (denom + eps)


def calculate_iou(gt_mask, pred_mask, eps=1e-6):
    gt = mask_to_bool(gt_mask)
    pr = mask_to_bool(pred_mask)
    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    if union == 0:
        return 1.0
    return inter / (union + eps)


def saliency_to_binary(vmap: np.ndarray, mode="otsu", threshold_value=0.3, percentile=85):
    """
    Convert normalized saliency map [0, 1] to bool mask.
    """
    vmap = normalize_map(vmap)
    u8 = np.clip(vmap * 255.0, 0, 255).astype(np.uint8)

    if mode == "fixed":
        thr = int(np.clip(threshold_value * 255.0, 0, 255))
        binary = (u8 >= thr)
    elif mode == "percentile":
        thr = np.percentile(vmap, percentile)
        binary = (vmap >= thr)
    elif mode == "otsu":
        _, th = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = th > 0
    else:
        raise ValueError(f"Unsupported threshold mode: {mode}")

    return binary.astype(bool)


def remove_small_components(mask_bool, min_area=10):
    """
    Keep only components >= min_area.
    """
    mask_u8 = mask_bool.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_bool, dtype=bool)
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area:
            out |= (labels == lab)
    return out


def keep_topk_components(mask_bool, keep_top_k=1, min_area=10):
    mask_u8 = mask_bool.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_bool.astype(bool)

    comps = []
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area:
            comps.append((area, lab))
    comps.sort(reverse=True)

    out = np.zeros_like(mask_bool, dtype=bool)
    for _, lab in comps[:keep_top_k]:
        out |= (labels == lab)
    return out


def component_confidence(vmap: np.ndarray, comp_mask: np.ndarray):
    vals = vmap[comp_mask > 0]
    if vals.size == 0:
        return 0.0
    return float(vals.mean())


def select_component_by_confidence(vmap: np.ndarray, mask_bool: np.ndarray, min_area=10):
    """
    Keep the connected component with the highest mean saliency.
    """
    mask_u8 = mask_bool.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_bool.astype(bool), 0.0

    best_score = -1.0
    best_lab = None
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == lab)
        score = component_confidence(vmap, comp)
        if score > best_score:
            best_score = score
            best_lab = lab

    if best_lab is None:
        return np.zeros_like(mask_bool, dtype=bool), 0.0

    return (labels == best_lab), best_score


def mask_to_box(mask_bool, pad_ratio=0.05, h=None, w=None):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    pad_x = int((x2 - x1 + 1) * pad_ratio)
    pad_y = int((y2 - y1 + 1) * pad_ratio)

    x1 -= pad_x
    x2 += pad_x
    y1 -= pad_y
    y2 += pad_y

    if w is not None:
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
    if h is not None:
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

    return [x1, y1, x2, y2]


def bbox_iou(box1, box2, eps=1e-6):
    if box1 is None or box2 is None:
        return 0.0
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    inter_w = max(0, xb - xa + 1)
    inter_h = max(0, yb - ya + 1)
    inter = inter_w * inter_h

    a1 = max(0, box1[2] - box1[0] + 1) * max(0, box1[3] - box1[1] + 1)
    a2 = max(0, box2[2] - box2[0] + 1) * max(0, box2[3] - box2[1] + 1)

    return inter / (a1 + a2 - inter + eps)


def infer_mask_name_from_image_name(image_name):
    # common convention: same stem, png
    stem = os.path.splitext(image_name)[0]
    return stem + ".png"


def build_component_mask(
    vmap_resized,
    threshold_mode="otsu",
    threshold_value=0.3,
    percentile=85,
    component_mode="confidence",
    keep_top_k=1,
    min_area_ratio=0.001,
):
    h, w = vmap_resized.shape[:2]
    min_area = max(1, int(h * w * min_area_ratio))

    binary = saliency_to_binary(
        vmap_resized,
        mode=threshold_mode,
        threshold_value=threshold_value,
        percentile=percentile,
    )

    binary = remove_small_components(binary, min_area=min_area)

    if component_mode == "largest":
        coarse = keep_topk_components(binary, keep_top_k=keep_top_k, min_area=min_area)
        comp_conf = component_confidence(vmap_resized, coarse.astype(np.uint8)) if coarse.any() else 0.0
    elif component_mode == "confidence":
        coarse, comp_conf = select_component_by_confidence(vmap_resized, binary, min_area=min_area)
    else:
        coarse = binary.astype(bool)
        comp_conf = component_confidence(vmap_resized, coarse.astype(np.uint8)) if coarse.any() else 0.0

    return coarse.astype(bool), float(comp_conf)


def stratified_sample(image_ids, mask_dir, n_samples=40, seed=42):
    """
    Stratify by lesion area quartiles using GT masks.
    """
    random.seed(seed)

    areas = []
    for img_id in image_ids:
        mask_name = infer_mask_name_from_image_name(img_id)
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            continue
        gt = read_mask(mask_path)
        area = int((gt > 0).sum())
        areas.append((img_id, area))

    if len(areas) == 0:
        return image_ids[: min(len(image_ids), n_samples)]

    areas.sort(key=lambda x: x[1])
    n = len(areas)
    bins = [areas[: max(1, n // 4)],
            areas[max(1, n // 4): max(2, n // 2)],
            areas[max(2, n // 2): max(3, 3 * n // 4)],
            areas[max(3, 3 * n // 4):]]

    selected = []
    per_bin = max(1, n_samples // 4)
    for b in bins:
        if len(b) == 0:
            continue
        pool = [x[0] for x in b]
        if len(pool) <= per_bin:
            selected.extend(pool)
        else:
            selected.extend(random.sample(pool, per_bin))

    # fill remaining
    selected = list(dict.fromkeys(selected))
    remaining = [x[0] for x in areas if x[0] not in selected]
    random.shuffle(remaining)
    selected.extend(remaining[: max(0, n_samples - len(selected))])

    return selected[: min(n_samples, len(selected))]


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_router_call(router, image_feature, top_k=None):
    fused = router.route(image_feature)
    debug_info = None
    try:
        debug_info = router.get_top_k_prompts(image_feature, k=top_k)
    except Exception:
        debug_info = None
    return fused, debug_info
def build_phrase_view_text(view_name, view_value):
    if view_value is None:
        return None

    view_value = str(view_value).strip()
    if len(view_value) == 0:
        return None

    if view_name == "modality":
        return f"A {view_value} image."
    elif view_name == "pathology":
        return f"A breast ultrasound image showing {view_value}."
    elif view_name == "morphology":
        return f"A breast ultrasound image showing a lesion with {view_value}."
    else:
        return view_value


def infer_patch_grid(model, image_tensor, patch_tokens):
    n_patches = int(patch_tokens.shape[1])

    patch_size = None
    try:
        patch_size = int(model.config.vision_config.patch_size)
    except Exception:
        patch_size = None

    if patch_size is not None:
        gh = int(image_tensor.shape[-2] // patch_size)
        gw = int(image_tensor.shape[-1] // patch_size)
        if gh * gw == n_patches:
            return gh, gw

    side = int(round(math.sqrt(n_patches)))
    if side * side != n_patches:
        raise ValueError(f"Cannot infer patch grid from {n_patches} patches.")
    return side, side



def project_patch_tokens_to_text_space(model, patch_tokens):
    """
    Project patch tokens from the vision hidden space into the joint image-text embedding space.
    patch_tokens: [B, N, Dv]
    returns: [B, N, Dt]
    """
    with torch.no_grad():
        if hasattr(model, "visual_projection"):
            projected = model.visual_projection(patch_tokens)
        elif hasattr(model, "vision_projection"):
            projected = model.vision_projection(patch_tokens)
        elif hasattr(model, "visual_proj"):
            projected = model.visual_proj(patch_tokens)
        else:
            raise AttributeError(
                "Model does not expose visual_projection / vision_projection / visual_proj. "
                "Cannot project patch tokens to joint embedding space."
            )

    projected = F.normalize(projected, dim=-1)
    return projected

def patch_text_similarity_map(patch_tokens, text_feature, gh, gw):
    """
    patch_tokens: [1, N, D]
    text_feature: [1, D]
    """
    if patch_tokens.shape[-1] != text_feature.shape[-1]:
        raise ValueError(
            f"Dimension mismatch: patch_tokens dim={patch_tokens.shape[-1]} "
            f"but text_feature dim={text_feature.shape[-1]}"
        )

    sim = torch.einsum("bnd,bd->bn", patch_tokens, text_feature).squeeze(0)
    sim = sim.view(gh, gw)
    sim = sim.detach().cpu().numpy().astype(np.float32)
    return normalize_map(sim)

def build_local_patch_text_map(
    top1_prompt,
    patch_tokens,
    image_tensor,
    model,
    tokenizer,
    device="cuda",
    w_modality=0.2,
    w_pathology=0.5,
    w_morphology=0.3,
):
    phrase_views = dict(top1_prompt.get("phrase_views", {}) or {})
    slots = dict(top1_prompt.get("slots", {}) or {})

    # fallback: derive phrase views from slots
    for k in ["modality", "pathology", "morphology"]:
        if not phrase_views.get(k) and slots.get(k):
            phrase_views[k] = slots[k]

    gh, gw = infer_patch_grid(model, image_tensor, patch_tokens)

    # Critical fix: project patch tokens into the same joint embedding space as text features
    patch_tokens = project_patch_tokens_to_text_space(model, patch_tokens)

    local_map = np.zeros((gh, gw), dtype=np.float32)
    used_views = []

    weights = {
        "modality": float(w_modality),
        "pathology": float(w_pathology),
        "morphology": float(w_morphology),
    }

    for view_name, view_weight in weights.items():
        if view_weight <= 0:
            continue

        view_text = build_phrase_view_text(view_name, phrase_views.get(view_name))
        if view_text is None:
            continue

        text_feature = get_text_features(
            model=model,
            tokenizer=tokenizer,
            text=view_text,
            device=device,
        )

        sim_map = patch_text_similarity_map(
            patch_tokens=patch_tokens,
            text_feature=text_feature,
            gh=gh,
            gw=gw,
        )

        local_map += view_weight * sim_map
        used_views.append({
            "name": view_name,
            "weight": view_weight,
            "text": view_text,
        })

    local_map = normalize_map(local_map)

    debug_info = {
        "used_views": used_views,
        "raw_phrase_views": phrase_views,
        "weights": weights,
    }
    return local_map, debug_info

def fuse_m2ib_and_local_map(m2ib_map, local_map, alpha_m2ib=0.6, alpha_local=0.4):
    m2ib_map = normalize_map(m2ib_map)
    local_map = normalize_map(local_map)
    fused = alpha_m2ib * m2ib_map + alpha_local * local_map
    return normalize_map(fused)

# =========================================================
# Model / prompt helpers
# =========================================================

def load_model_and_processor(args):
    print("Loading models ...")
    finetuned = getattr(args, "finetuned", False)

    if args.model_name == "BiomedCLIP" and finetuned:
        model = AutoModel.from_pretrained(
            "./saliency_maps/model",
            trust_remote_code=True
        ).to(args.device)
        processor = AutoProcessor.from_pretrained(
            "chuhac/BiomedCLIP-vit-bert-hf",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "chuhac/BiomedCLIP-vit-bert-hf",
            trust_remote_code=True
        )

    elif args.model_name == "BiomedCLIP" and not finetuned:
        model = AutoModel.from_pretrained(
            "chuhac/BiomedCLIP-vit-bert-hf",
            trust_remote_code=True
        ).to(args.device)
        processor = AutoProcessor.from_pretrained(
            "chuhac/BiomedCLIP-vit-bert-hf",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "chuhac/BiomedCLIP-vit-bert-hf",
            trust_remote_code=True
        )

    else:
        raise ValueError(f"Unsupported combination: model_name={args.model_name}, finetuned={finetuned}")

    model.eval()
    return model, processor, tokenizer

def load_prompt_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_text_feature_for_image_id(
    image_id,
    args,
    model,
    tokenizer,
    router=None,
):
    """
    Priority:
    1) if --use-router and router exists: route by image feature
    2) else if --reproduce and prompt json exists: use image-specific prompt
    3) else if --text is given: use global custom prompt
    """
    router_info = None

    if args.use_router and router is not None:
        # router needs image feature
        return None, {"_defer_router": True}

    if args.reproduce:
        if not os.path.exists(args.prompt_json):
            raise FileNotFoundError(f"Prompt json not found: {args.prompt_json}")
        prompt_json = load_prompt_json(args.prompt_json)
        if image_id not in prompt_json:
            raise KeyError(f"{image_id} not found in prompt json: {args.prompt_json}")
        text_prompt = prompt_json[image_id]
        text_feature = get_text_features(model, tokenizer, text_prompt, device=args.device)
        router_info = {
            "mode": "reproduce_json",
            "image_id": image_id,
            "prompt": text_prompt,
        }
        return text_feature, router_info

    if args.text is not None and len(args.text.strip()) > 0:
        text_feature = get_text_features(model, tokenizer, args.text, device=args.device)
        router_info = {
            "mode": "global_text",
            "image_id": image_id,
            "prompt": args.text,
        }
        return text_feature, router_info

    raise ValueError("No prompt source available. Use --use-router, or --reproduce with --prompt-json, or --text.")


def compute_saliency_map(
    img_bgr,
    image_id,
    args,
    model,
    processor,
    tokenizer,
    router=None,
):
    debug_meta = {
        "image_id": image_id,
        "router_info": None,
        "success": True,
    }

    # 鍥惧儚鐗瑰緛 + 鍥惧儚tensor
    need_patch_tokens = bool(getattr(args, "use_local_fusion", False) and getattr(args, "use_router", False))

    img_outputs = get_img_features(
        model,
        processor,
        img_bgr,
        device=args.device,
        return_img_tensor=True,
        return_patch_tokens=need_patch_tokens,
    )

    if need_patch_tokens:
        image_feature, image_tensor, patch_tokens = img_outputs
    else:
        image_feature, image_tensor = img_outputs
        patch_tokens = None

    prompt_json_path = ""
    if getattr(args, "prompt_json", ""):
        prompt_json_path = args.prompt_json
    elif getattr(args, "json_path", ""):
        prompt_json_path = args.json_path

    # -------------------------------------------------
    # router 妯″紡锛氫紶 fused text feature锛屽苟璁剧疆 precomputed_text_feature=True
    # -------------------------------------------------
    if getattr(args, "use_router", False) and router is not None:
        fused_text_feature, router_info = safe_router_call(
            router,
            image_feature,
            top_k=getattr(args, "top_k", None)
        )

        debug_meta["router_info"] = {
            "mode": "router",
            "topk": router_info
        }

        vmap = vision_heatmap_iba(
            fused_text_feature,
            image_tensor,
            model,
            args.vlayer,
            args.vbeta,
            args.vvar,
            ensemble=getattr(args, "ensemble", False),
            progbar=False,
            precomputed_text_feature=True,
        )

        # Positive saliency (normalize first)
        vmap_pos = np.asarray(vmap, dtype=np.float32)
        vmap_pos = normalize_map(vmap_pos)

        # Contrastive saliency: subtract negative/background activation
        if getattr(args, 'use_contrastive', False):
            neg_text_feature = get_text_features(model, tokenizer, args.negative_prompt, device=args.device)
            vmap_neg = vision_heatmap_iba(
                neg_text_feature,
                image_tensor,
                model,
                args.vlayer,
                args.vbeta,
                args.vvar,
                ensemble=getattr(args, "ensemble", False),
                progbar=False,
                precomputed_text_feature=True,
            )
            vmap_neg = np.asarray(vmap_neg, dtype=np.float32)
            vmap_neg = normalize_map(vmap_neg)
            vmap = vmap_pos - (float(args.contrastive_lambda) * vmap_neg)
            vmap = np.clip(vmap, 0, None)
            vmap = normalize_map(vmap)
        else:
            vmap = vmap_pos
    
        # -------------------------------------------------
        # local patch-text fusion (optional)
        # -------------------------------------------------
        if getattr(args, "use_local_fusion", False):
            try:
                top1_list = router.get_top_k_prompts(image_feature, k=1)
                top1_prompt = top1_list[0] if len(top1_list) > 0 else {}

                local_map, local_debug = build_local_patch_text_map(
                    top1_prompt=top1_prompt,
                    patch_tokens=patch_tokens,
                    image_tensor=image_tensor,
                    model=model,
                    tokenizer=tokenizer,
                    device=args.device,
                    w_modality=args.pv_weight_modality,
                    w_pathology=args.pv_weight_pathology,
                    w_morphology=args.pv_weight_morphology,
                )

                local_map = cv2.resize(
                    local_map,
                    (vmap.shape[1], vmap.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

                vmap = fuse_m2ib_and_local_map(
                    m2ib_map=vmap,
                    local_map=local_map,
                    alpha_m2ib=args.fusion_alpha_m2ib,
                    alpha_local=args.fusion_alpha_local,
                )

                debug_meta["local_fusion"] = {
                    "enabled": True,
                    "top1_prompt": top1_prompt,
                    "local_debug": local_debug,
                    "fusion_alpha_m2ib": args.fusion_alpha_m2ib,
                    "fusion_alpha_local": args.fusion_alpha_local,
                }
            except Exception as e:
                print(f"[WARN] local fusion failed for {image_id}: {e}")
                debug_meta["local_fusion"] = {
                    "enabled": False,
                    "error": str(e),
                }
        else:
            debug_meta["local_fusion"] = {
                "enabled": False
            }

    # -------------------------------------------------
    # reproduce / 鏅€氭枃鏈ā寮忥細浼� token ids锛屽苟璁剧疆 precomputed_text_feature=False
    # -------------------------------------------------
    else:
        if getattr(args, "reproduce", False):
            if not prompt_json_path:
                raise ValueError("reproduce 妯″紡涓嬬己灏� --json-path 鎴� --prompt-json")
            with open(prompt_json_path, "r", encoding="utf-8") as f:
                prompt_json = json.load(f)
            if image_id not in prompt_json:
                raise KeyError(f"{image_id} 涓嶅湪 prompt json 涓�: {prompt_json_path}")
            text = prompt_json[image_id]
            debug_meta["router_info"] = {
                "mode": "reproduce_json",
                "prompt": text,
            }
        else:
            text = args.text
            debug_meta["router_info"] = {
                "mode": "global_text",
                "prompt": text,
            }

        text_ids = torch.tensor(
            [tokenizer.encode(text, add_special_tokens=True)]
        ).to(args.device)

        vmap = vision_heatmap_iba(
            text_ids,                # text_t: token ids
            image_tensor,            # image_t
            model,                   # model
            args.vlayer,             # layer_idx
            args.vbeta,              # beta
            args.vvar,               # var
            ensemble=getattr(args, "ensemble", False),
            progbar=False,
            precomputed_text_feature=False,
        )

        # Positive saliency (normalize first)
        vmap_pos = np.asarray(vmap, dtype=np.float32)
        vmap_pos = normalize_map(vmap_pos)

        if getattr(args, 'use_contrastive', False):
            neg_text_feature = get_text_features(model, tokenizer, args.negative_prompt, device=args.device)
            vmap_neg = vision_heatmap_iba(
                neg_text_feature,
                image_tensor,
                model,
                args.vlayer,
                args.vbeta,
                args.vvar,
                ensemble=getattr(args, "ensemble", False),
                progbar=False,
                precomputed_text_feature=True,
            )
            vmap_neg = np.asarray(vmap_neg, dtype=np.float32)
            vmap_neg = normalize_map(vmap_neg)
            vmap = vmap_pos - (float(args.contrastive_lambda) * vmap_neg)
            vmap = np.clip(vmap, 0, None)
            vmap = normalize_map(vmap)
        else:
            vmap = vmap_pos
    
    vmap = np.asarray(vmap, dtype=np.float32)
    vmap = normalize_map(vmap)

    return vmap, debug_meta


# =========================================================
# Evaluation
# =========================================================

def evaluate_single_sample(
    image_id,
    image_dir,
    mask_dir,
    args,
    model,
    processor,
    tokenizer,
    router=None,
):
    """
    Returns dict metrics for one sample.
    """
    img_path = os.path.join(image_dir, image_id)
    mask_path = os.path.join(mask_dir, infer_mask_name_from_image_name(image_id))

    img = read_image(img_path)
    gt_mask = read_mask(mask_path)
    h, w = gt_mask.shape[:2]

    vmap_224, debug_meta = compute_saliency_map(
        img_bgr=img,
        image_id=image_id,
        args=args,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        router=router,
    )

    vmap_resized = resize_saliency(vmap_224, w, h)

    raw_binary_fixed = saliency_to_binary(
        vmap_resized,
        mode="fixed",
        threshold_value=args.eval_fixed_threshold,
    )
    raw_dice = calculate_dice_coefficient(gt_mask, raw_binary_fixed)
    raw_iou = calculate_iou(gt_mask, raw_binary_fixed)

    coarse_mask, comp_conf = build_component_mask(
        vmap_resized=vmap_resized,
        threshold_mode=args.threshold_mode,
        threshold_value=args.threshold_value,
        percentile=args.threshold_percentile,
        component_mode=args.component_mode,
        keep_top_k=args.keep_top_k,
        min_area_ratio=args.min_area_ratio,
    )

    coarse_dice = calculate_dice_coefficient(gt_mask, coarse_mask)
    coarse_iou = calculate_iou(gt_mask, coarse_mask)

    pred_box = mask_to_box(coarse_mask, pad_ratio=args.box_pad_ratio, h=h, w=w)
    gt_box = mask_to_box(mask_to_bool(gt_mask), pad_ratio=0.0, h=h, w=w)
    box_iou_val = bbox_iou(pred_box, gt_box)

    # final hyperopt target: closer to downstream than pure raw threshold dice
    score = 0.7 * coarse_dice + 0.3 * box_iou_val

    result = {
        "image_id": image_id,
        "raw_dice_fixed": float(raw_dice),
        "raw_iou_fixed": float(raw_iou),
        "coarse_dice": float(coarse_dice),
        "coarse_iou": float(coarse_iou),
        "box_iou": float(box_iou_val),
        "component_confidence": float(comp_conf),
        "score": float(score),
        "router_info": debug_meta.get("router_info"),
    }
    return result


def evaluate_config(
    image_ids,
    image_dir,
    mask_dir,
    args,
    model,
    processor,
    tokenizer,
    router=None,
    verbose=False,
):
    rows = []
    iterator = tqdm(image_ids, disable=not verbose, desc="Eval config")
    for image_id in iterator:
        try:
            row = evaluate_single_sample(
                image_id=image_id,
                image_dir=image_dir,
                mask_dir=mask_dir,
                args=args,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                router=router,
            )
            rows.append(row)
        except Exception as e:
            rows.append({
                "image_id": image_id,
                "raw_dice_fixed": 0.0,
                "raw_iou_fixed": 0.0,
                "coarse_dice": 0.0,
                "coarse_iou": 0.0,
                "box_iou": 0.0,
                "component_confidence": 0.0,
                "score": 0.0,
                "router_info": {"error": str(e)},
            })

    if len(rows) == 0:
        return {
            "mean_score": 0.0,
            "mean_coarse_dice": 0.0,
            "mean_box_iou": 0.0,
            "details": [],
        }

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "router_info"} for r in rows])
    return {
        "mean_score": float(df["score"].mean()),
        "mean_coarse_dice": float(df["coarse_dice"].mean()),
        "mean_box_iou": float(df["box_iou"].mean()),
        "details": rows,
    }


# =========================================================
# Hyperopt
# =========================================================

def maybe_build_router(args, model, tokenizer):
    if not getattr(args, "use_router", False):
        return None

    if PromptRouter is None:
        raise ImportError("PromptRouter could not be imported, but --use-router was set.")

    from pathlib import Path
    import os

    prompt_dir = getattr(args, "router_prompt_dir", "")
    if not prompt_dir:
        raise ValueError("--use-router 宸插紑鍚紝浣嗘病鏈夋彁渚� --router-prompt-dir")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(prompt_dir):
        prompt_dir = os.path.abspath(os.path.join(script_dir, prompt_dir))

    if not os.path.isdir(prompt_dir):
        raise FileNotFoundError(f"router prompt 鐩綍涓嶅瓨鍦�: {prompt_dir}")

    prompt_bank_paths = sorted([
        str(p) for p in Path(prompt_dir).glob("*.json")
        if p.is_file()
    ])

    if len(prompt_bank_paths) == 0:
        raise ValueError(f"router prompt 鐩綍閲屾病鏈� json 鏂囦欢: {prompt_dir}")

    print(f"[INFO] Building PromptRouter from {len(prompt_bank_paths)} json files in: {prompt_dir}")

    router = PromptRouter(
        model=model,
        tokenizer=tokenizer,
        prompt_bank_paths=prompt_bank_paths,
        top_k=args.top_k,
        temperature=args.router_temperature,
        device=args.device,
    )
    return router


def hyper_opt(args, model, processor, tokenizer):
    if args.val_path is None or len(args.val_path.strip()) == 0:
        raise ValueError("--hyper-opt requires --val-path")

    val_image_dir = args.val_path
    val_mask_dir = args.val_mask_path if args.val_mask_path else args.gt_path

    if val_mask_dir is None:
        raise ValueError("--hyper-opt requires val masks. Set --val-mask-path or --gt-path")

    val_image_ids = list_images(val_image_dir)
    sampled_ids = stratified_sample(
        image_ids=val_image_ids,
        mask_dir=val_mask_dir,
        n_samples=args.hyperopt_samples,
        seed=args.seed,
    )

    router = maybe_build_router(args, model, tokenizer)

    # search space
    betas = [float(x) for x in args.hyperopt_betas.split(",")]
    vars_ = [float(x) for x in args.hyperopt_vars.split(",")]
    layers = [int(x) for x in args.hyperopt_layers.split(",")]

    top_ks = [args.top_k]
    temps = [args.router_temperature]

    if args.use_router:
        top_ks = [int(x) for x in args.hyperopt_topks.split(",")]
        temps = [float(x) for x in args.hyperopt_temperatures.split(",")]

    results = []
    best = None
    best_score = -1.0

    search_space = list(itertools.product(betas, vars_, layers, top_ks, temps))
    print(f"[INFO] Hyperopt samples: {len(sampled_ids)}")
    print(f"[INFO] Search configs: {len(search_space)}")

    for (beta, var, layer, top_k, temp) in tqdm(search_space, desc="Hyperopt"):
        args.vbeta = beta
        args.vvar = var
        args.vlayer = layer

        if router is not None:
            # best effort: update runtime fields if router supports them
            if hasattr(router, "top_k"):
                router.top_k = top_k
            if hasattr(router, "temperature"):
                router.temperature = temp

        res = evaluate_config(
            image_ids=sampled_ids,
            image_dir=val_image_dir,
            mask_dir=val_mask_dir,
            args=args,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            router=router,
            verbose=False,
        )

        row = {
            "vbeta": beta,
            "vvar": var,
            "vlayer": layer,
            "top_k": top_k,
            "router_temperature": temp,
            "mean_score": res["mean_score"],
            "mean_coarse_dice": res["mean_coarse_dice"],
            "mean_box_iou": res["mean_box_iou"],
            "num_samples": len(sampled_ids),
        }
        results.append(row)

        if row["mean_score"] > best_score:
            best_score = row["mean_score"]
            best = dict(row)

    res_df = pd.DataFrame(results)
    ensure_dir(args.output_path)
    res_df.to_csv(os.path.join(args.output_path, "hyperopt_results.csv"), index=False)

    if best is None:
        raise RuntimeError("Hyperopt failed to find a valid configuration.")

    save_json(os.path.join(args.output_path, "best_config.json"), best)

    args.vbeta = float(best["vbeta"])
    args.vvar = float(best["vvar"])
    args.vlayer = int(best["vlayer"])
    if args.use_router:
        args.top_k = int(best["top_k"])
        args.router_temperature = float(best["router_temperature"])

    print("[INFO] Best hyperopt config:", best)
    return best


# =========================================================
# Main inference loop
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    # core io
    parser.add_argument("--input-path", type=str, required=True, help="Input images folder")
    parser.add_argument("--output-path", type=str, required=True, help="Output saliency folder")
    parser.add_argument("--gt-path", type=str, default="", help="GT masks folder for metrics")
    parser.add_argument("--val-path", type=str, default="", help="Validation images folder for hyperopt")
    parser.add_argument("--val-mask-path", type=str, default="", help="Validation masks folder for hyperopt")
    parser.add_argument("--run-name", type=str, default="", help="Append subfolder under output-path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    # model
    parser.add_argument("--model-name", type=str, default="BiomedCLIP", choices=["BiomedCLIP"])
    parser.add_argument("--model-path", type=str, default="./model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # saliency params
    parser.add_argument("--vbeta", type=float, default=0.5)
    parser.add_argument("--vvar", type=float, default=1.0)
    parser.add_argument("--vlayer", type=int, default=8)

    # prompt sources
    parser.add_argument("--reproduce", action="store_true", help="Use image-specific prompt json")
    parser.add_argument("--prompt-json", type=str, default="./text_prompts/breast_tumors_testing.json")
    parser.add_argument("--text", type=str, default="", help="Single global prompt")
    parser.add_argument("--use-contrastive", action="store_true", 
                        help="Enable contrastive saliency (pos - lambda*neg).")
    parser.add_argument("--negative-prompt", type=str, 
                        default="Normal healthy tissue, background, and anatomical structures without any lesions",
                        help="Background/normal text prompt for negative saliency.")
    parser.add_argument("--contrastive-lambda", type=float, default=0.4,
                        help="Weight for subtracting negative saliency.")
    parser.add_argument("--use-router", action="store_true", help="Use prompt router")
    parser.add_argument("--router-prompt-dir", type=str, default="./text_prompts/breast_lesion_prompts")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--router-temperature", type=float, default=0.05)

    # thresholding / coarse-mask generation
    parser.add_argument("--threshold-mode", type=str, default="otsu", choices=["fixed", "otsu", "percentile"])
    parser.add_argument("--threshold-value", type=float, default=0.3)
    parser.add_argument("--threshold-percentile", type=float, default=85.0)
    parser.add_argument("--component-mode", type=str, default="confidence", choices=["confidence", "largest", "none"])
    parser.add_argument("--keep-top-k", type=int, default=1)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--box-pad-ratio", type=float, default=0.05)

    # eval
    parser.add_argument("--eval-fixed-threshold", type=float, default=0.3)
    parser.add_argument("--save-stage-metrics", action="store_true")
    parser.add_argument("--save-router-debug", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--finetuned", action="store_true")
    # hyperopt
    parser.add_argument("--hyper-opt", action="store_true")
    parser.add_argument("--hyperopt-samples", type=int, default=40)
    parser.add_argument("--hyperopt-betas", type=str, default="0.1,0.5,1.0,2.0")
    parser.add_argument("--hyperopt-vars", type=str, default="0.1,0.5,1.0,2.0")
    parser.add_argument("--hyperopt-layers", type=str, default="7,8,9")
    parser.add_argument("--hyperopt-topks", type=str, default="1,2,3")
    parser.add_argument("--hyperopt-temperatures", type=str, default="0.03,0.05,0.07,0.10")
    parser.add_argument(
    "--hf-model-id",
    type=str,
    default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
    # misc
    parser.add_argument("--task", type=str, default="breast_tumors")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-local-fusion", action="store_true")
    parser.add_argument("--fusion-alpha-m2ib", type=float, default=0.6)
    parser.add_argument("--fusion-alpha-local", type=float, default=0.4)

    parser.add_argument("--pv-weight-modality", type=float, default=0.2)
    parser.add_argument("--pv-weight-pathology", type=float, default=0.5)
    parser.add_argument("--pv-weight-morphology", type=float, default=0.3)
    args = parser.parse_args()
    set_seed(args.seed)
    if args.use_local_fusion and (not args.use_router):
        print("[WARN] --use-local-fusion 闇€瑕� --use-router锛屽凡鑷姩鍏抽棴 local fusion.")
        args.use_local_fusion = False
    if args.run_name:
        args.output_path = os.path.join(args.output_path, args.run_name)

    ensure_dir(args.output_path)
    if args.save_router_debug:
        ensure_dir(os.path.join(args.output_path, "router_debug"))
    if args.save_preview:
        ensure_dir(os.path.join(args.output_path, "preview"))

    if args.reproduce and args.use_router:
        print("[WARN] Both --reproduce and --use-router are set. Router will take precedence.")

    model, processor, tokenizer = load_model_and_processor(args)

    # hyperopt first
    if args.hyper_opt:
        hyper_opt(args, model, processor, tokenizer)

    router = maybe_build_router(args, model, tokenizer)

    image_ids = list_images(args.input_path)
    existing_outputs = set() if args.overwrite else set(os.listdir(args.output_path))

    stage_rows = []
    start_time = time.time()

    for image_id in tqdm(image_ids, desc="Generating saliency"):
        out_name = image_id
        out_path = os.path.join(args.output_path, out_name)

        # skip existing only when overwrite is False
        if (not args.overwrite) and (out_name in existing_outputs):
            continue

        img_path = os.path.join(args.input_path, image_id)
        img = read_image(img_path)

        try:
            vmap_224, debug_meta = compute_saliency_map(
                img_bgr=img,
                image_id=image_id,
                args=args,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                router=router,
            )

            h, w = img.shape[:2]
            vmap_resized = resize_saliency(vmap_224, w, h)
            save_saliency(out_path, vmap_resized)

            # optional metrics if GT provided
            row = {
                "image_id": image_id,
                "saved_path": out_path,
                "success": 1,
                "error": "",
            }

            if args.gt_path and os.path.exists(args.gt_path):
                gt_path = os.path.join(args.gt_path, infer_mask_name_from_image_name(image_id))
                if os.path.exists(gt_path):
                    gt_mask = read_mask(gt_path)

                    raw_binary_fixed = saliency_to_binary(
                        vmap_resized,
                        mode="fixed",
                        threshold_value=args.eval_fixed_threshold,
                    )
                    raw_dice = calculate_dice_coefficient(gt_mask, raw_binary_fixed)
                    raw_iou = calculate_iou(gt_mask, raw_binary_fixed)

                    coarse_mask, comp_conf = build_component_mask(
                        vmap_resized=vmap_resized,
                        threshold_mode=args.threshold_mode,
                        threshold_value=args.threshold_value,
                        percentile=args.threshold_percentile,
                        component_mode=args.component_mode,
                        keep_top_k=args.keep_top_k,
                        min_area_ratio=args.min_area_ratio,
                    )
                    coarse_dice = calculate_dice_coefficient(gt_mask, coarse_mask)
                    coarse_iou = calculate_iou(gt_mask, coarse_mask)

                    pred_box = mask_to_box(
                        coarse_mask,
                        pad_ratio=args.box_pad_ratio,
                        h=gt_mask.shape[0],
                        w=gt_mask.shape[1]
                    )
                    gt_box = mask_to_box(mask_to_bool(gt_mask), pad_ratio=0.0, h=gt_mask.shape[0], w=gt_mask.shape[1])
                    box_iou_val = bbox_iou(pred_box, gt_box)

                    row.update({
                        "raw_dice_fixed": float(raw_dice),
                        "raw_iou_fixed": float(raw_iou),
                        "coarse_dice": float(coarse_dice),
                        "coarse_iou": float(coarse_iou),
                        "box_iou": float(box_iou_val),
                        "component_confidence": float(comp_conf),
                    })

            if debug_meta.get("router_info") is not None:
                row["router_mode"] = debug_meta["router_info"].get("mode", "router_or_prompt")
            else:
                row["router_mode"] = "unknown"

            stage_rows.append(row)

            if args.save_router_debug and debug_meta.get("router_info") is not None:
                save_json(
                    os.path.join(args.output_path, "router_debug", os.path.splitext(image_id)[0] + ".json"),
                    debug_meta["router_info"],
                )

            if args.save_preview:
                try:
                    preview_path = os.path.join(args.output_path, "preview", image_id)
                    save_saliency(preview_path, vmap_resized)
                except Exception:
                    pass

        except Exception as e:
            stage_rows.append({
                "image_id": image_id,
                "saved_path": "",
                "success": 0,
                "error": str(e),
            })
            print(f"[ERROR] {image_id}: {e}")

    if args.save_stage_metrics or len(stage_rows) > 0:
        pd.DataFrame(stage_rows).to_csv(
            os.path.join(args.output_path, "stage_metrics.csv"),
            index=False
        )

    elapsed = time.time() - start_time
    print(f"[INFO] Done. Output: {args.output_path}")
    print(f"[INFO] Processed {len(image_ids)} images in {elapsed / 60.0:.2f} minutes.")


if __name__ == "__main__":
    main()