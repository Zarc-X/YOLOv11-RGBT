#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

E2_WEIGHTS=${E2_WEIGHTS:-runs/GAIIC2024_e1e6_auto/e1e6-r01/e2e4/e2-cont80-img896/weights/best.pt}
E3_WEIGHTS=${E3_WEIGHTS:-runs/GAIIC2024_e1e6_auto/mp-e3-20260627-095652/e2e4/e3-cont80-img768/weights/best.pt}

OUT_DIR=${OUT_DIR:-runs/GAIIC2024_submit_chain/$(date +%Y%m%d-%H%M%S)}
DEVICE=${DEVICE:-0}
BATCH=${BATCH:-8}

CONF=${CONF:-0.001}
IOU=${IOU:-0.6}
MAX_DET=${MAX_DET:-500}

WBF_IOU=${WBF_IOU:-0.65}
WBF_SCORE=${WBF_SCORE:-0.001}
WBF_MAX_DET=${WBF_MAX_DET:-500}

SHRINK_SCORE=${SHRINK_SCORE:-0.01}
SHRINK_MAX_DET=${SHRINK_MAX_DET:-120}

mkdir -p "$OUT_DIR/preds" "$OUT_DIR/wbf" "$OUT_DIR/submissions"

echo "[INFO] OUT_DIR=$OUT_DIR"

python export_gaiic_coco.py \
  --weights "$E2_WEIGHTS" --split test --imgsz 896 --conf "$CONF" --iou "$IOU" --max-det "$MAX_DET" \
  --device "$DEVICE" --batch "$BATCH" \
  --out "$OUT_DIR/preds/e2_base.json"

python export_gaiic_coco.py \
  --weights "$E2_WEIGHTS" --split test --imgsz 896 --conf "$CONF" --iou "$IOU" --max-det "$MAX_DET" \
  --device "$DEVICE" --batch "$BATCH" --augment \
  --out "$OUT_DIR/preds/e2_tta.json"

python export_gaiic_coco.py \
  --weights "$E3_WEIGHTS" --split test --imgsz 768 --conf "$CONF" --iou "$IOU" --max-det "$MAX_DET" \
  --device "$DEVICE" --batch "$BATCH" \
  --out "$OUT_DIR/preds/e3_base.json"

python export_gaiic_coco.py \
  --weights "$E3_WEIGHTS" --split test --imgsz 768 --conf "$CONF" --iou "$IOU" --max-det "$MAX_DET" \
  --device "$DEVICE" --batch "$BATCH" --augment \
  --out "$OUT_DIR/preds/e3_tta.json"

python tools/weighted_box_fusion.py \
  --inputs "$OUT_DIR/preds/e2_base.json" "$OUT_DIR/preds/e3_base.json" \
  --weights 0.5 0.5 \
  --iou-thr "$WBF_IOU" --score-thr "$WBF_SCORE" --max-det "$WBF_MAX_DET" \
  --output "$OUT_DIR/wbf/wbf_e2e3_50_50.json"

python tools/weighted_box_fusion.py \
  --inputs "$OUT_DIR/preds/e2_base.json" "$OUT_DIR/preds/e3_base.json" \
  --weights 0.4 0.6 \
  --iou-thr "$WBF_IOU" --score-thr "$WBF_SCORE" --max-det "$WBF_MAX_DET" \
  --output "$OUT_DIR/wbf/wbf_e2e3_40_60.json"

python tools/weighted_box_fusion.py \
  --inputs "$OUT_DIR/preds/e2_base.json" "$OUT_DIR/preds/e3_base.json" "$OUT_DIR/preds/e3_tta.json" \
  --weights 0.3 0.4 0.3 \
  --iou-thr "$WBF_IOU" --score-thr "$WBF_SCORE" --max-det "$WBF_MAX_DET" \
  --output "$OUT_DIR/wbf/wbf_e2e3e3tta_30_40_30.json"

for src in \
  "$OUT_DIR/preds/e2_base.json" \
  "$OUT_DIR/preds/e2_tta.json" \
  "$OUT_DIR/preds/e3_base.json" \
  "$OUT_DIR/preds/e3_tta.json" \
  "$OUT_DIR/wbf/wbf_e2e3_50_50.json" \
  "$OUT_DIR/wbf/wbf_e2e3_40_60.json" \
  "$OUT_DIR/wbf/wbf_e2e3e3tta_30_40_30.json"; do
  name=$(basename "$src" .json)
  python tools/shrink_submission.py \
    --input "$src" \
    --output "$OUT_DIR/submissions/${name}_shrink.json" \
    --score-thr "$SHRINK_SCORE" \
    --max-det-per-image "$SHRINK_MAX_DET"
done

export OUT_DIR
python - <<'PY'
import csv, json
import os
from pathlib import Path
out = Path(os.environ["OUT_DIR"])
subs = sorted((out / "submissions").glob("*.json"))
rows = []
for p in subs:
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows.append({
        "file": str(p),
        "size_mb": round(p.stat().st_size / 1024 / 1024, 4),
        "pred_records": len(data),
        "images_with_pred": len({x['image_id'] for x in data}),
    })
manifest = out / "submission_manifest.csv"
with manifest.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file", "size_mb", "pred_records", "images_with_pred"])
    w.writeheader()
    w.writerows(rows)
print(f"manifest {manifest}")
for r in rows:
    print(r)
PY

echo "[DONE] submit chain outputs under: $OUT_DIR"
