#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

# Lightweight submit chain (2~3 models) with optional TTA and WBF.
# Required: MODEL1_WEIGHTS
MODEL1_WEIGHTS="${MODEL1_WEIGHTS:-}"
MODEL1_IMGSZ="${MODEL1_IMGSZ:-768}"
MODEL1_WEIGHT="${MODEL1_WEIGHT:-1.0}"

MODEL2_WEIGHTS="${MODEL2_WEIGHTS:-}"
MODEL2_IMGSZ="${MODEL2_IMGSZ:-768}"
MODEL2_WEIGHT="${MODEL2_WEIGHT:-1.0}"

MODEL3_WEIGHTS="${MODEL3_WEIGHTS:-}"
MODEL3_IMGSZ="${MODEL3_IMGSZ:-896}"
MODEL3_WEIGHT="${MODEL3_WEIGHT:-1.0}"

OUT_DIR="${OUT_DIR:-runs/GAIIC2024_lightweight_submit/$(date +%Y%m%d-%H%M%S)}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-8}"

CONF="${CONF:-0.001}"
IOU="${IOU:-0.6}"
MAX_DET="${MAX_DET:-500}"

EXPORT_TTA="${EXPORT_TTA:-1}"
TTA_WEIGHT_SCALE="${TTA_WEIGHT_SCALE:-0.5}"

WBF_IOU="${WBF_IOU:-0.65}"
WBF_SCORE="${WBF_SCORE:-0.001}"
WBF_MAX_DET="${WBF_MAX_DET:-500}"

SHRINK_SCORE="${SHRINK_SCORE:-0.01}"
SHRINK_MAX_DET="${SHRINK_MAX_DET:-120}"

if [[ -z "$MODEL1_WEIGHTS" ]]; then
  echo "[ERROR] MODEL1_WEIGHTS is required." >&2
  exit 1
fi

mkdir -p "$OUT_DIR/preds" "$OUT_DIR/wbf" "$OUT_DIR/submissions"

declare -a INPUTS=()
declare -a WEIGHTS=()

auto_export() {
  local idx="$1"
  local w="$2"
  local imgsz="$3"
  local weight_scale="$4"

  local base_json="$OUT_DIR/preds/m${idx}_base.json"
  python export_gaiic_coco.py \
    --weights "$w" --split test --imgsz "$imgsz" --conf "$CONF" --iou "$IOU" --max-det "$MAX_DET" \
    --device "$DEVICE" --batch "$BATCH" \
    --out "$base_json"

  INPUTS+=("$base_json")
  WEIGHTS+=("$weight_scale")

  if [[ "$EXPORT_TTA" == "1" ]]; then
    local tta_json="$OUT_DIR/preds/m${idx}_tta.json"
    python export_gaiic_coco.py \
      --weights "$w" --split test --imgsz "$imgsz" --conf "$CONF" --iou "$IOU" --max-det "$MAX_DET" \
      --device "$DEVICE" --batch "$BATCH" --augment \
      --out "$tta_json"

    local tta_weight
    tta_weight=$(python - <<PY
base=float("$weight_scale")
scale=float("$TTA_WEIGHT_SCALE")
print(base*scale)
PY
)
    INPUTS+=("$tta_json")
    WEIGHTS+=("$tta_weight")
  fi
}

auto_export 1 "$MODEL1_WEIGHTS" "$MODEL1_IMGSZ" "$MODEL1_WEIGHT"

if [[ -n "$MODEL2_WEIGHTS" ]]; then
  auto_export 2 "$MODEL2_WEIGHTS" "$MODEL2_IMGSZ" "$MODEL2_WEIGHT"
fi

if [[ -n "$MODEL3_WEIGHTS" ]]; then
  auto_export 3 "$MODEL3_WEIGHTS" "$MODEL3_IMGSZ" "$MODEL3_WEIGHT"
fi

echo "[INFO] Inputs for WBF: ${#INPUTS[@]}"
python tools/weighted_box_fusion.py \
  --inputs "${INPUTS[@]}" \
  --weights "${WEIGHTS[@]}" \
  --iou-thr "$WBF_IOU" --score-thr "$WBF_SCORE" --max-det "$WBF_MAX_DET" \
  --output "$OUT_DIR/wbf/wbf_lightweight.json"

for src in "$OUT_DIR"/preds/*.json "$OUT_DIR/wbf/wbf_lightweight.json"; do
  name=$(basename "$src" .json)
  python tools/shrink_submission.py \
    --input "$src" \
    --output "$OUT_DIR/submissions/${name}_shrink.json" \
    --score-thr "$SHRINK_SCORE" \
    --max-det-per-image "$SHRINK_MAX_DET"
done

export OUT_DIR
python - <<'PY'
import csv
import json
import os
from pathlib import Path

out = Path(os.environ["OUT_DIR"])
subs = sorted((out / "submissions").glob("*.json"))
rows = []
for p in subs:
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows.append(
        {
            "file": str(p),
            "size_mb": round(p.stat().st_size / 1024 / 1024, 4),
            "pred_records": len(data),
            "images_with_pred": len({x["image_id"] for x in data}),
        }
    )

manifest = out / "submission_manifest.csv"
with manifest.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "size_mb", "pred_records", "images_with_pred"])
    writer.writeheader()
    writer.writerows(rows)

print(f"manifest {manifest}")
for row in rows:
    print(row)
PY

echo "[DONE] lightweight submit chain outputs: $OUT_DIR"
