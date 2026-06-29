#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def mean_valid(x: np.ndarray) -> float:
    x = x[x > -1]
    return float(np.mean(x)) if x.size else float('nan')


def evaluate_per_class(gt_json: Path, dt_json: Path, max_det: int = 100) -> dict:
    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(dt_json))
    ev = COCOeval(coco_gt, coco_dt, 'bbox')
    ev.params.maxDets = [1, 10, max_det]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    p = ev.eval['precision']  # [T, R, K, A, M]
    m_idx = 2  # maxDet index for 100
    a_idx = 0  # area all

    out = {
        'ap50_95': mean_valid(p[:, :, :, a_idx, m_idx]),
        'ap50': mean_valid(p[0, :, :, a_idx, m_idx]),
        'per_class': {},
    }
    for k, cid in enumerate(ev.params.catIds):
        cat_name = coco_gt.loadCats([cid])[0]['name']
        out['per_class'][cat_name] = mean_valid(p[:, :, k, a_idx, m_idx])

    # quick score distribution summary
    with dt_json.open('r', encoding='utf-8') as f:
        dt = json.load(f)
    scores = np.array([float(x.get('score', 0.0)) for x in dt], dtype=np.float32)
    if scores.size:
        out['score_p50'] = float(np.quantile(scores, 0.5))
        out['score_p90'] = float(np.quantile(scores, 0.9))
        out['score_p95'] = float(np.quantile(scores, 0.95))
    out['num_preds'] = int(scores.size)
    out['images_with_pred'] = int(len({x['image_id'] for x in dt}))

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Per-class COCO AP analysis for one or more detection json files')
    p.add_argument('--gt', type=Path, required=True, help='Ground-truth COCO json')
    p.add_argument('--dt', nargs='+', type=Path, required=True, help='Detection json paths')
    p.add_argument('--max-det', type=int, default=100)
    p.add_argument('--out', type=Path, default=Path('runs/analysis/per_class_ap.json'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for dt in args.dt:
        if not dt.exists():
            print(f'[SKIP] missing: {dt}')
            continue
        print(f'\n=== {dt} ===')
        rec = evaluate_per_class(args.gt, dt, max_det=args.max_det)
        all_results[str(dt)] = rec
        print(f"AP50-95={rec['ap50_95']:.6f} AP50={rec['ap50']:.6f} preds={rec['num_preds']} imgs={rec['images_with_pred']}")
        print('Per-class AP50-95:')
        for k, v in rec['per_class'].items():
            print(f'  {k}: {v:.6f}')

    args.out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'\n[DONE] {args.out}')


if __name__ == '__main__':
    main()
