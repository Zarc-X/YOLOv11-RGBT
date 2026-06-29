#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shrink COCO detection JSON for strict upload size limits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-thr", type=float, default=0.01)
    parser.add_argument("--max-det-per-image", type=int, default=120)
    parser.add_argument("--bbox-decimals", type=int, default=1)
    parser.add_argument("--score-decimals", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        detections = json.load(f)

    grouped: dict[object, list[dict]] = defaultdict(list)
    for det in detections:
        score = float(det["score"])
        if score < args.score_thr:
            continue
        grouped[det["image_id"]].append(det)

    compact: list[dict] = []
    for image_id, dets in grouped.items():
        dets.sort(key=lambda d: float(d["score"]), reverse=True)
        for det in dets[: args.max_det_per_image]:
            compact.append(
                {
                    "image_id": image_id,
                    "category_id": int(det["category_id"]),
                    "bbox": [round(float(v), args.bbox_decimals) for v in det["bbox"]],
                    "score": round(float(det["score"]), args.score_decimals),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"Wrote {len(compact)} detections to {args.output} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
