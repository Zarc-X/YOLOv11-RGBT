#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
from pycocotools.coco import COCO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard negative car detections from COCO-format predictions")
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path("/root/autodl-tmp/GAIIC2024-赛道1-目标检测任务/val/val.json"),
        help="Ground-truth COCO json path",
    )
    parser.add_argument("--dt", type=Path, required=True, help="Detection json path in COCO result format")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("/root/autodl-tmp/GAIIC2024-赛道1-目标检测任务/val/rgb"),
        help="Image root directory for crop export",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/analysis/car_hardnegatives"),
        help="Output directory",
    )
    parser.add_argument("--category-id", type=int, default=1, help="Target category id, 1 means car")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Only keep detections >= this score")
    parser.add_argument(
        "--iou-bg-thr",
        type=float,
        default=0.1,
        help="Keep detections whose max IoU with any GT box is below this threshold",
    )
    parser.add_argument("--max-samples", type=int, default=500, help="Maximum number of hard negatives to export")
    parser.add_argument("--crop-margin", type=float, default=0.1, help="Crop margin ratio around bbox")
    parser.add_argument("--pairs-rgb", type=str, default="rgb", help="RGB token in image path")
    parser.add_argument("--pairs-ir", type=str, default="tir", help="IR token in image path")
    parser.add_argument("--no-crops", action="store_true", help="Only export CSV, do not save crop images")
    return parser.parse_args()


def iou_xywh(b1: list[float], b2: list[float]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter <= 0:
        return 0.0
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def main() -> None:
    args = parse_args()

    if not args.gt.exists():
        raise FileNotFoundError(f"GT json not found: {args.gt}")
    if not args.dt.exists():
        raise FileNotFoundError(f"DT json not found: {args.dt}")
    if not args.no_crops and not args.image_root.exists():
        raise FileNotFoundError(f"Image root not found: {args.image_root}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    if not args.no_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)
    crops_ir_dir = out_dir / "crops_ir"
    if not args.no_crops:
        crops_ir_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(args.gt))
    images = {im["id"]: im for im in coco.dataset.get("images", [])}

    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)
    gt_by_img: dict[int, list[dict]] = {}
    for ann in anns:
        if ann.get("iscrowd", 0):
            continue
        gt_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    with args.dt.open("r", encoding="utf-8") as f:
        dt = json.load(f)

    hard: list[dict] = []
    for pred in dt:
        if int(pred.get("category_id", -1)) != args.category_id:
            continue
        score = float(pred.get("score", 0.0))
        if score < args.score_thr:
            continue

        image_id = int(pred["image_id"])
        pb = [float(x) for x in pred["bbox"]]
        gt_list = gt_by_img.get(image_id, [])

        max_iou = 0.0
        for g in gt_list:
            iv = iou_xywh(pb, [float(x) for x in g["bbox"]])
            if iv > max_iou:
                max_iou = iv

        if max_iou < args.iou_bg_thr:
            hard.append(
                {
                    "image_id": image_id,
                    "score": score,
                    "bbox": pb,
                    "area": pb[2] * pb[3],
                    "max_iou_any_gt": max_iou,
                    "is_empty_image": 1 if len(gt_list) == 0 else 0,
                }
            )

    hard.sort(key=lambda x: x["score"], reverse=True)
    hard = hard[: max(1, args.max_samples)]

    csv_path = out_dir / "hardnegatives.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "image_id",
                "file_name",
                "score",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "area",
                "max_iou_any_gt",
                "is_empty_image",
                "crop_path",
                    "crop_ir_path",
            ]
        )

        for i, h in enumerate(hard, start=1):
            image_id = int(h["image_id"])
            im = images.get(image_id, {})
            file_name = str(im.get("file_name", f"{image_id:05d}.jpg"))

            crop_rel = ""
            crop_ir_rel = ""
            if not args.no_crops:
                img_path = args.image_root / file_name
                if img_path.exists():
                    arr = cv2.imread(str(img_path))
                    if arr is not None:
                        ih, iw = arr.shape[:2]
                        x, y, w, hgt = h["bbox"]
                        mx = float(args.crop_margin) * w
                        my = float(args.crop_margin) * hgt
                        x1 = clamp(int(round(x - mx)), 0, iw - 1)
                        y1 = clamp(int(round(y - my)), 0, ih - 1)
                        x2 = clamp(int(round(x + w + mx)), 1, iw)
                        y2 = clamp(int(round(y + hgt + my)), 1, ih)
                        if x2 > x1 and y2 > y1:
                            crop = arr[y1:y2, x1:x2]
                            crop_name = f"{i:04d}_img{image_id:05d}_s{h['score']:.3f}.jpg"
                            crop_path = crops_dir / crop_name
                            cv2.imwrite(str(crop_path), crop)
                            crop_rel = str(crop_path.relative_to(out_dir))

                            # Try to export paired IR crop with the same window.
                            ir_path = Path(str(img_path).replace(args.pairs_rgb, args.pairs_ir))
                            if ir_path.exists():
                                ir_arr = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)
                                if ir_arr is not None and ir_arr.ndim >= 2:
                                    ir_h, ir_w = ir_arr.shape[:2]
                                    ix1 = clamp(x1, 0, ir_w - 1)
                                    iy1 = clamp(y1, 0, ir_h - 1)
                                    ix2 = clamp(x2, 1, ir_w)
                                    iy2 = clamp(y2, 1, ir_h)
                                    if ix2 > ix1 and iy2 > iy1:
                                        ir_crop = ir_arr[iy1:iy2, ix1:ix2]
                                        ir_crop_name = f"{i:04d}_img{image_id:05d}_s{h['score']:.3f}.png"
                                        ir_crop_path = crops_ir_dir / ir_crop_name
                                        cv2.imwrite(str(ir_crop_path), ir_crop)
                                        crop_ir_rel = str(ir_crop_path.relative_to(out_dir))

            writer.writerow(
                [
                    i,
                    image_id,
                    file_name,
                    f"{h['score']:.6f}",
                    f"{h['bbox'][0]:.3f}",
                    f"{h['bbox'][1]:.3f}",
                    f"{h['bbox'][2]:.3f}",
                    f"{h['bbox'][3]:.3f}",
                    f"{h['area']:.3f}",
                    f"{h['max_iou_any_gt']:.6f}",
                    h["is_empty_image"],
                    crop_rel,
                    crop_ir_rel,
                ]
            )

    print(f"hardnegatives_csv {csv_path}")
    print(f"hardnegatives_count {len(hard)}")
    if not args.no_crops:
        print(f"hardnegatives_crops_dir {crops_dir}")
        print(f"hardnegatives_crops_ir_dir {crops_ir_dir}")


if __name__ == "__main__":
    main()
