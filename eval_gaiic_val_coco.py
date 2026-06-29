#!/usr/bin/env python3
import argparse
import json
import tempfile
import os
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GAIIC val COCO json with COCOeval")
    parser.add_argument(
        "--gt",
        type=str,
        default="/root/autodl-tmp/GAIIC2024-赛道1-目标检测任务/val/val.json",
        help="Ground-truth COCO json path",
    )
    parser.add_argument(
        "--dt",
        type=str,
        required=True,
        help="Detection result json path (COCO results format)",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="COCOeval maxDets last value, typical 100",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=None,
        help="Optional score threshold before eval",
    )
    parser.add_argument(
        "--topk-per-image",
        type=int,
        default=None,
        help="Optional top-k predictions per image before eval",
    )
    return parser.parse_args()


def filter_dt(dt, score_thr=None, topk_per_image=None):
    if score_thr is not None:
        dt = [x for x in dt if x.get("score", 0.0) >= score_thr]

    if topk_per_image is not None:
        by = {}
        for x in dt:
            by.setdefault(x["image_id"], []).append(x)
        filtered = []
        for _, arr in by.items():
            arr.sort(key=lambda z: z["score"], reverse=True)
            filtered.extend(arr[:topk_per_image])
        dt = filtered

    return dt


def mean_valid(arr):
    arr = arr[arr > -1]
    return float(np.mean(arr)) if arr.size else float("nan")


def main():
    args = parse_args()

    with open(args.dt, "r", encoding="utf-8") as f:
        dt = json.load(f)

    dt = filter_dt(dt, args.score_thr, args.topk_per_image)

    coco_gt = COCO(args.gt)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(dt, f, ensure_ascii=False)
        tmp_path = f.name

    coco_dt = coco_gt.loadRes(tmp_path)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = [1, 10, args.max_det]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    precision = evaluator.eval["precision"]

    m_index = 2
    ap5095 = mean_valid(precision[:, :, :, 0, m_index])
    ap50 = mean_valid(precision[0, :, :, 0, m_index])

    print(f"AP5095_recomputed {ap5095:.6f}")
    print(f"AP50_recomputed {ap50:.6f}")

    cat_ids = coco_gt.getCatIds()
    cats = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    for k, cat_id in enumerate(cat_ids):
        ap = mean_valid(precision[:, :, k, 0, m_index])
        print(f"AP5095 {cats[cat_id]} ({cat_id}): {ap:.4f}")

    os.remove(tmp_path)


if __name__ == "__main__":
    main()
