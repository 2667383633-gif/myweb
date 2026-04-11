#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate breast_tumors_training.json for MedCLIP-SAMv2
using the paper/repo-style class-specific P3 prompts.

Usage:
    python make_breast_tumors_training_json.py \
        --train-images data/breast_tumors/train_images \
        --output saliency_maps/text_prompts/breast_tumors_training.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_repo_prompts(repo_root: Path):
    """
    Import the exact breast P3 prompt pools from the released repo.
    """
    sys.path.insert(0, str(repo_root))
    try:
        from saliency_maps.text_prompts import (  # type: ignore
            benign_breast_tumor_P3_prompts,
            malignant_breast_tumor_P3_prompts,
        )
    except Exception as e:
        raise ImportError(
            "无法从仓库导入 saliency_maps.text_prompts。\n"
            "请确保：\n"
            "1) 你在 MedCLIP-SAMv2 仓库根目录下运行，或\n"
            "2) 用 --repo-root 指向 MedCLIP-SAMv2 仓库根目录。\n"
            f"原始错误: {e}"
        ) from e

    return benign_breast_tumor_P3_prompts, malignant_breast_tumor_P3_prompts


def deterministic_pick(key: str, prompts: List[str]) -> str:
    """
    Deterministically pick one prompt from a prompt pool using SHA1.
    This keeps the mapping reproducible across runs.
    """
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(prompts)
    return prompts[idx]


def infer_label(path: Path) -> Optional[str]:
    """
    Infer label from filename or parent folders.

    Supports common BUSI-style names such as:
      benign (1).png
      malignant (23).png
      benign/xxx.png
      malignant/xxx.png

    Returns:
      'benign', 'malignant', 'normal', or None
    """
    text = str(path).lower().replace("\\", "/")

    if "benign" in text:
        return "benign"
    if "malignant" in text:
        return "malignant"
    if "normal" in text:
        return "normal"
    return None


def is_image_file(path: Path) -> bool:
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return path.is_file() and path.suffix.lower() in valid_exts


def collect_images(train_images_dir: Path) -> List[Path]:
    """
    Recursively collect image files and skip obvious mask files.
    """
    images = []
    for p in sorted(train_images_dir.rglob("*")):
        if not is_image_file(p):
            continue
        stem_lower = p.stem.lower()
        if stem_lower.endswith("_mask") or "_mask." in p.name.lower():
            continue
        images.append(p)
    return images


def build_json(
    train_images_dir: Path,
    benign_prompts: List[str],
    malignant_prompts: List[str],
    strict: bool = False,
) -> Dict[str, str]:
    """
    Build {filename: prompt} mapping.
    """
    result: Dict[str, str] = {}
    skipped_normal = []
    skipped_unknown = []

    image_paths = collect_images(train_images_dir)

    for img_path in image_paths:
        label = infer_label(img_path)

        if label == "normal":
            skipped_normal.append(img_path.name)
            continue

        if label == "benign":
            prompt = deterministic_pick(img_path.name, benign_prompts)
        elif label == "malignant":
            prompt = deterministic_pick(img_path.name, malignant_prompts)
        else:
            skipped_unknown.append(img_path.name)
            if strict:
                raise ValueError(
                    f"无法从文件名/路径推断类别: {img_path}\n"
                    "请检查命名中是否包含 benign 或 malignant。"
                )
            continue

        key = img_path.name  # keep basename only, same style as existing prompt JSONs
        if key in result:
            raise ValueError(
                f"检测到重复文件名: {key}\n"
                "当前脚本按 basename 作为 JSON key；请先确保 train_images 内无重名文件。"
            )
        result[key] = prompt

    print(f"[INFO] total image files found: {len(image_paths)}")
    print(f"[INFO] prompts generated     : {len(result)}")
    print(f"[INFO] skipped normal        : {len(skipped_normal)}")
    print(f"[INFO] skipped unknown       : {len(skipped_unknown)}")

    if skipped_unknown:
        print("\n[WARN] 以下文件未能识别类别，已跳过：")
        for name in skipped_unknown[:20]:
            print(f"  - {name}")
        if len(skipped_unknown) > 20:
            print(f"  ... 共 {len(skipped_unknown)} 个")

    return dict(sorted(result.items(), key=lambda x: x[0]))


def main():
    parser = argparse.ArgumentParser(
        description="Generate breast_tumors_training.json for MedCLIP-SAMv2"
    )
    parser.add_argument(
        "--train-images",
        type=Path,
        required=True,
        help="Path to data/breast_tumors/train_images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path, e.g. saliency_maps/text_prompts/breast_tumors_training.json",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(".").resolve(),
        help="Path to MedCLIP-SAMv2 repo root (default: current directory)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any image label cannot be inferred from path/name",
    )

    args = parser.parse_args()

    train_images_dir = args.train_images.resolve()
    repo_root = args.repo_root.resolve()
    output_path = args.output.resolve()

    if not train_images_dir.exists():
        raise FileNotFoundError(f"train_images 不存在: {train_images_dir}")
    if not train_images_dir.is_dir():
        raise NotADirectoryError(f"train_images 不是目录: {train_images_dir}")

    benign_prompts, malignant_prompts = load_repo_prompts(repo_root)

    mapping = build_json(
        train_images_dir=train_images_dir,
        benign_prompts=benign_prompts,
        malignant_prompts=malignant_prompts,
        strict=args.strict,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] saved to: {output_path}")


if __name__ == "__main__":
    main()