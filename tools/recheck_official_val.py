#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate multiple weights with official-style COCO val protocol")
    parser.add_argument(
        "--weights",
        nargs="+",
        required=True,
        help="One or more weight paths (.pt)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/root/autodl-tmp/GAIIC2024-赛道1-目标检测任务"),
        help="GAIIC dataset root",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="GT json path for official val eval. Default: <dataset-root>/val/val.json",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--use-simotm", type=str, default="RGBT")
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--pairs-rgb", type=str, default="rgb")
    parser.add_argument("--pairs-ir", type=str, default="tir")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/GAIIC2024_archscan"),
        help="Output directory for leaderboard files",
    )
    return parser.parse_args()


def mean_valid(arr: np.ndarray) -> float:
    arr = arr[arr > -1]
    return float(np.mean(arr)) if arr.size else float("nan")


def make_temp_data_yaml(dataset_root: Path) -> Path:
    data = {
        "path": str(dataset_root),
        "train": "train/rgb",
        "val": "val/rgb",
        "nc": 5,
        "names": ["car", "truck", "bus", "van", "freight_car"],
    }
    fd, p = tempfile.mkstemp(prefix="official_val_", suffix=".yaml")
    Path(p).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    Path(p).chmod(0o644)
    return Path(p)


def quantile(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, max(0, int(p * len(sorted_vals)) - 1))
    return sorted_vals[i]


def evaluate_dt(gt_path: Path, dt_path: Path, max_det: int) -> tuple[float, float]:
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(dt_path))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = [1, 10, max_det]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    precision = evaluator.eval["precision"]
    m_index = 2
    ap5095 = mean_valid(precision[:, :, :, 0, m_index])
    ap50 = mean_valid(precision[0, :, :, 0, m_index])
    return ap5095, ap50


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root.resolve()
    gt = args.gt.resolve() if args.gt else (dataset_root / "val" / "val.json").resolve()
    if not gt.exists():
        raise FileNotFoundError(f"GT json not found: {gt}")

    rows: list[dict] = []

    for w in args.weights:
        weights = Path(w).resolve()
        if not weights.exists():
            print(f"[SKIP] weights not found: {weights}")
            continue

        run_dir = weights.parent.parent
        tmp_project = run_dir / "_tmp_official_val"
        tmp_name = "pred_export"
        tmp_dir = tmp_project / tmp_name
        temp_yaml = make_temp_data_yaml(dataset_root)

        try:
            model = YOLO(str(weights))
            model.val(
                data=str(temp_yaml),
                split="val",
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=max(300, args.max_det),
                batch=args.batch,
                workers=args.workers,
                device=args.device,
                use_simotm=args.use_simotm,
                channels=args.channels,
                pairs_rgb_ir=[args.pairs_rgb, args.pairs_ir],
                save_json=True,
                plots=False,
                save=False,
                verbose=False,
                project=str(tmp_project),
                name=tmp_name,
                exist_ok=True,
            )

            pred_json = tmp_dir / "predictions.json"
            if not pred_json.exists():
                raise FileNotFoundError(f"predictions.json missing: {pred_json}")

            dt = json.loads(pred_json.read_text(encoding="utf-8"))
            by = defaultdict(int)
            for x in dt:
                by[x["image_id"]] += 1
            vals = sorted(by.values())

            ap5095, ap50 = evaluate_dt(gt, pred_json, args.max_det)

            rec = {
                "weights": str(weights),
                "run_dir": str(run_dir),
                "official_val_map50_95": ap5095,
                "official_val_map50": ap50,
                "pred_records": len(dt),
                "images_with_pred": len(by),
                "det_per_image_mean": (sum(vals) / len(vals)) if vals else float("nan"),
                "det_per_image_p50": quantile(vals, 0.5),
                "det_per_image_p90": quantile(vals, 0.9),
                "det_per_image_p95": quantile(vals, 0.95),
                "det_per_image_p99": quantile(vals, 0.99),
            }
            rows.append(rec)

            one = run_dir / "archscan_public_val_official.json"
            one.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                f"[OK] {weights.name} | AP50-95={ap5095:.6f} AP50={ap50:.6f} "
                f"pred={len(dt)} imgs={len(by)}"
            )
        finally:
            temp_yaml.unlink(missing_ok=True)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if tmp_project.exists() and not any(tmp_project.iterdir()):
                tmp_project.rmdir()

    rows.sort(key=lambda x: x.get("official_val_map50_95", float("nan")), reverse=True)

    jsonl_path = out_dir / "official_val_recheck.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_path = out_dir / "official_val_leaderboard.csv"
    headers = [
        "rank",
        "weights",
        "run_dir",
        "official_val_map50_95",
        "official_val_map50",
        "pred_records",
        "images_with_pred",
        "det_per_image_mean",
        "det_per_image_p50",
        "det_per_image_p90",
        "det_per_image_p95",
        "det_per_image_p99",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i, r in enumerate(rows, start=1):
            w.writerow(
                [
                    i,
                    r["weights"],
                    r["run_dir"],
                    f"{r['official_val_map50_95']:.6f}" if not math.isnan(r["official_val_map50_95"]) else "nan",
                    f"{r['official_val_map50']:.6f}" if not math.isnan(r["official_val_map50"]) else "nan",
                    r["pred_records"],
                    r["images_with_pred"],
                    f"{r['det_per_image_mean']:.4f}" if not math.isnan(r["det_per_image_mean"]) else "nan",
                    r["det_per_image_p50"],
                    r["det_per_image_p90"],
                    r["det_per_image_p95"],
                    r["det_per_image_p99"],
                ]
            )

    print(f"[DONE] jsonl: {jsonl_path}")
    print(f"[DONE] csv: {csv_path}")


if __name__ == "__main__":
    main()
