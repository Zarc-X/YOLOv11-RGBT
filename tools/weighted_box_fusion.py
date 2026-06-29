#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def coco_to_xyxy(bbox: list[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox]
    return np.asarray([x, y, x + max(0.0, w), y + max(0.0, h)], dtype=float)


def xyxy_to_coco(box: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list json: {path}")
    return data


def save_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def fuse_group(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[tuple[np.ndarray, float]]:
    if len(boxes) == 0:
        return []

    valid = (
        np.isfinite(boxes).all(axis=1)
        & np.isfinite(scores)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
        & (scores > 0)
    )
    boxes = boxes[valid]
    scores = scores[valid]
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    used = np.zeros(len(boxes), dtype=bool)
    fused: list[tuple[np.ndarray, float]] = []

    for i in range(len(boxes)):
        if used[i]:
            continue
        ious = box_iou(boxes[i], boxes)
        cluster_idx = np.where((ious >= iou_thr) & (~used))[0]
        if len(cluster_idx) == 0:
            cluster_idx = np.asarray([i])

        cluster_boxes = boxes[cluster_idx]
        cluster_scores = scores[cluster_idx]
        weights = cluster_scores / np.maximum(cluster_scores.sum(), 1e-9)
        fused_box = (cluster_boxes * weights[:, None]).sum(axis=0)

        agreement_score = cluster_scores.mean() + 0.05 * (len(cluster_idx) - 1)
        fused_score = float(min(max(cluster_scores.max(), agreement_score), 1.0))

        fused.append((fused_box, fused_score))
        used[cluster_idx] = True

    fused.sort(key=lambda item: item[1], reverse=True)
    return fused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge COCO detection JSON files with Weighted Box Fusion.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-thr", type=float, default=0.65)
    parser.add_argument("--score-thr", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights = args.weights or [1.0] * len(args.inputs)
    if len(weights) != len(args.inputs):
        raise ValueError("--weights length must match --inputs length")

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for model_idx, (path, model_weight) in enumerate(zip(args.inputs, weights)):
        dets = load_json(path)
        for det in dets:
            score = float(det["score"]) * model_weight
            if score < args.score_thr:
                continue
            grouped[(det["image_id"], int(det["category_id"]))].append(
                {
                    "bbox": det["bbox"],
                    "score": min(score, 1.0),
                    "model_idx": model_idx,
                }
            )

    per_image: dict[object, list[dict]] = defaultdict(list)
    for (image_id, category_id), dets in grouped.items():
        boxes = np.asarray([coco_to_xyxy(det["bbox"]) for det in dets], dtype=float)
        scores = np.asarray([det["score"] for det in dets], dtype=float)
        for box, score in fuse_group(boxes, scores, args.iou_thr):
            per_image[image_id].append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [round(v, 1) for v in xyxy_to_coco(box)],
                    "score": round(float(score), 4),
                }
            )

    merged: list[dict] = []
    for image_id, dets in per_image.items():
        dets.sort(key=lambda item: item["score"], reverse=True)
        merged.extend(dets[: args.max_det])

    merged.sort(key=lambda d: (str(d["image_id"]), int(d["category_id"]), -float(d["score"])))
    save_json(args.output, merged)
    print(f"Wrote {len(merged)} WBF detections to {args.output}")


if __name__ == "__main__":
    main()
