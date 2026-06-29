#!/usr/bin/env python3
"""Convert GAIIC2024 COCO JSON annotations to YOLO txt labels.

This script writes label files next to both RGB and TIR images, which matches
YOLOv11-RGBT's recommended layout for paired training.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert COCO JSON to YOLO labels for GAIIC RGB/TIR dataset")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/root/autodl-tmp/GAIIC2024-赛道1-目标检测任务"),
        help="Path to dataset root containing train/ and val/",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to convert, e.g. train val",
    )
    parser.add_argument(
        "--rgb-dir-name",
        type=str,
        default="rgb",
        help="RGB directory name under each split",
    )
    parser.add_argument(
        "--tir-dir-name",
        type=str,
        default="tir",
        help="TIR directory name under each split",
    )
    parser.add_argument(
        "--copy-to-tir",
        action="store_true",
        default=True,
        help="Also write same labels next to TIR images (default: true)",
    )
    parser.add_argument(
        "--no-copy-to-tir",
        dest="copy_to_tir",
        action="store_false",
        help="Disable writing labels to TIR side",
    )
    return parser.parse_args()


def coco_to_yolo_bbox(bbox: List[float], width: float, height: float) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    xc = (x + w / 2.0) / width
    yc = (y + h / 2.0) / height
    wn = w / width
    hn = h / height
    return xc, yc, wn, hn


def clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def write_labels_for_split(dataset_root: Path, split: str, rgb_dir_name: str, tir_dir_name: str, copy_to_tir: bool) -> None:
    split_dir = dataset_root / split
    json_path = split_dir / f"{split}.json"
    rgb_dir = split_dir / rgb_dir_name
    tir_dir = split_dir / tir_dir_name

    if not json_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {json_path}")
    if not rgb_dir.exists():
        raise FileNotFoundError(f"Missing RGB directory: {rgb_dir}")
    if copy_to_tir and not tir_dir.exists():
        raise FileNotFoundError(f"Missing TIR directory: {tir_dir}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images: Dict[int, dict] = {img["id"]: img for img in data.get("images", [])}

    cat_ids = sorted(c["id"] for c in data.get("categories", []))
    cat_id_to_zero = {cid: i for i, cid in enumerate(cat_ids)}

    image_to_lines: Dict[int, List[str]] = {img_id: [] for img_id in images.keys()}

    skipped = 0
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        img = images.get(img_id)
        if img is None:
            skipped += 1
            continue

        width = float(img["width"])
        height = float(img["height"])
        if width <= 0 or height <= 0:
            skipped += 1
            continue

        cid = ann["category_id"]
        if cid not in cat_id_to_zero:
            skipped += 1
            continue

        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue

        xc, yc, wn, hn = coco_to_yolo_bbox([x, y, w, h], width, height)
        xc = clamp01(xc)
        yc = clamp01(yc)
        wn = clamp01(wn)
        hn = clamp01(hn)

        line = f"{cat_id_to_zero[cid]} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
        image_to_lines[img_id].append(line)

    wrote_rgb = 0
    wrote_tir = 0
    missing_rgb_img = 0
    missing_tir_img = 0

    for img_id, img in images.items():
        stem = Path(img["file_name"]).stem
        rgb_img = rgb_dir / img["file_name"]
        rgb_txt = rgb_dir / f"{stem}.txt"

        if not rgb_img.exists():
            missing_rgb_img += 1
            continue

        lines = image_to_lines.get(img_id, [])
        rgb_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        wrote_rgb += 1

        if copy_to_tir:
            tir_img = tir_dir / img["file_name"]
            tir_txt = tir_dir / f"{stem}.txt"
            if tir_img.exists():
                tir_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                wrote_tir += 1
            else:
                missing_tir_img += 1

    print(f"[{split}] images={len(images)} anns={len(data.get('annotations', []))} categories={len(cat_ids)}")
    print(f"[{split}] wrote RGB labels: {wrote_rgb}, missing RGB images: {missing_rgb_img}")
    if copy_to_tir:
        print(f"[{split}] wrote TIR labels: {wrote_tir}, missing TIR images: {missing_tir_img}")
    if skipped:
        print(f"[{split}] skipped annotations: {skipped}")


def main() -> None:
    args = parse_args()
    for split in args.splits:
        write_labels_for_split(args.dataset_root, split, args.rgb_dir_name, args.tir_dir_name, args.copy_to_tir)


if __name__ == "__main__":
    main()
