#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

# End-to-end pipeline for large-model adaptation:
# 1) run E5/E6 with auto modal inheritance + tuning
# 2) export base and optional TTA predictions
# 3) run WBF grid (complementary fusion)
# 4) shrink all candidates and write manifest

# Required
MAIN_WEIGHTS="${MAIN_WEIGHTS:-}"
if [[ -z "$MAIN_WEIGHTS" ]]; then
  echo "[ERROR] MAIN_WEIGHTS is required." >&2
  exit 1
fi

# Optional complementary branch (can be current best 4C branch)
AUX_WEIGHTS="${AUX_WEIGHTS:-}"

# Runtime
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-2}"
SEED="${SEED:-42}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/GAIIC2024-赛道1-目标检测任务}"

# Projects
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
BASE_PROJECT="${BASE_PROJECT:-runs/GAIIC2024_large_model_pipeline}"
RUN_ROOT="$BASE_PROJECT/$RUN_ID"
TRAIN_ROOT="$RUN_ROOT/train"
EXPORT_ROOT="$RUN_ROOT/export"
WBF_ROOT="$RUN_ROOT/wbf"
SUB_ROOT="$RUN_ROOT/submissions"
mkdir -p "$TRAIN_ROOT" "$EXPORT_ROOT" "$WBF_ROOT" "$SUB_ROOT"

# E5/E6 tuning (large-model friendly defaults)
MAIN_IMGSZ="${MAIN_IMGSZ:-768}"
MAIN_E5_FOLDS="${MAIN_E5_FOLDS:-2}"
MAIN_E5_ROUNDS="${MAIN_E5_ROUNDS:-1}"
MAIN_E5_BASE_EPOCHS="${MAIN_E5_BASE_EPOCHS:-15}"
MAIN_E5_ONLINE_EPOCHS="${MAIN_E5_ONLINE_EPOCHS:-8}"
MAIN_E5_LR0="${MAIN_E5_LR0:-0.0016}"
MAIN_E5_LRF="${MAIN_E5_LRF:-0.01}"
MAIN_E5_MOMENTUM="${MAIN_E5_MOMENTUM:-0.937}"
MAIN_E5_WEIGHT_DECAY="${MAIN_E5_WEIGHT_DECAY:-0.0005}"
MAIN_E5_PATIENCE="${MAIN_E5_PATIENCE:-16}"

MAIN_RUN_E6="${MAIN_RUN_E6:-1}"
MAIN_E6_EPOCHS="${MAIN_E6_EPOCHS:-24}"
MAIN_E6_IMGSZ="${MAIN_E6_IMGSZ:-768}"
MAIN_E6_LR0="${MAIN_E6_LR0:-0.0012}"
MAIN_E6_LRF="${MAIN_E6_LRF:-0.01}"
MAIN_E6_MOMENTUM="${MAIN_E6_MOMENTUM:-0.937}"
MAIN_E6_WEIGHT_DECAY="${MAIN_E6_WEIGHT_DECAY:-0.0005}"
MAIN_E6_PATIENCE="${MAIN_E6_PATIENCE:-16}"

# Export
CONF="${CONF:-0.001}"
IOU="${IOU:-0.6}"
MAX_DET="${MAX_DET:-500}"
EXPORT_TTA="${EXPORT_TTA:-1}"
MAIN_TTA_WEIGHT_SCALE="${MAIN_TTA_WEIGHT_SCALE:-0.5}"
AUX_TTA_WEIGHT_SCALE="${AUX_TTA_WEIGHT_SCALE:-0.5}"

# WBF + shrink grid
WBF_IOUS="${WBF_IOUS:-0.62,0.65,0.68}"
WBF_SCORES="${WBF_SCORES:-0.001,0.003}"
SHRINK_SCORES="${SHRINK_SCORES:-0.008,0.01,0.012}"
SHRINK_MAX_DET="${SHRINK_MAX_DET:-120}"

# Base fusion weights
MAIN_WEIGHT="${MAIN_WEIGHT:-1.0}"
AUX_WEIGHT="${AUX_WEIGHT:-0.85}"

echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] MAIN_WEIGHTS=$MAIN_WEIGHTS"
echo "[INFO] AUX_WEIGHTS=${AUX_WEIGHTS:-<none>}"

# 1) Large-model adapted E5/E6
TRAIN_NAME="main-e5e6"
TRAIN_SUMMARY="$TRAIN_ROOT/$TRAIN_NAME/pipeline_summary.json"
RUN_E6_FLAG=()
if [[ "$MAIN_RUN_E6" == "1" ]]; then
  RUN_E6_FLAG+=(--run-e6)
fi

python tools/run_e5_e6_from_weight.py \
  --initial-weights "$MAIN_WEIGHTS" \
  --dataset-root "$DATASET_ROOT" \
  --imgsz "$MAIN_IMGSZ" \
  --device "$DEVICE" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --e5-folds "$MAIN_E5_FOLDS" \
  --e5-rounds "$MAIN_E5_ROUNDS" \
  --e5-base-epochs "$MAIN_E5_BASE_EPOCHS" \
  --e5-online-epochs "$MAIN_E5_ONLINE_EPOCHS" \
  --e5-lr0 "$MAIN_E5_LR0" \
  --e5-lrf "$MAIN_E5_LRF" \
  --e5-momentum "$MAIN_E5_MOMENTUM" \
  --e5-weight-decay "$MAIN_E5_WEIGHT_DECAY" \
  --e5-patience "$MAIN_E5_PATIENCE" \
  --e6-epochs "$MAIN_E6_EPOCHS" \
  --e6-imgsz "$MAIN_E6_IMGSZ" \
  --e6-lr0 "$MAIN_E6_LR0" \
  --e6-lrf "$MAIN_E6_LRF" \
  --e6-momentum "$MAIN_E6_MOMENTUM" \
  --e6-weight-decay "$MAIN_E6_WEIGHT_DECAY" \
  --e6-patience "$MAIN_E6_PATIENCE" \
  --project "$TRAIN_ROOT" \
  --name "$TRAIN_NAME" \
  --strict-run-root \
  "${RUN_E6_FLAG[@]}"

if [[ ! -f "$TRAIN_SUMMARY" ]]; then
  echo "[ERROR] Missing training summary: $TRAIN_SUMMARY" >&2
  exit 1
fi

MAIN_FINAL_WEIGHTS=$(python - <<'PY' "$TRAIN_SUMMARY"
import json, sys
obj = json.loads(open(sys.argv[1], 'r', encoding='utf-8').read())
e6 = obj.get('e6', {})
if e6.get('status') == 'completed' and e6.get('weights'):
    print(e6['weights'])
else:
    print(obj.get('e5', {}).get('best_weights', ''))
PY
)

if [[ -z "$MAIN_FINAL_WEIGHTS" || ! -f "$MAIN_FINAL_WEIGHTS" ]]; then
  echo "[ERROR] Failed to resolve MAIN_FINAL_WEIGHTS from $TRAIN_SUMMARY" >&2
  exit 1
fi

echo "[INFO] MAIN_FINAL_WEIGHTS=$MAIN_FINAL_WEIGHTS"

# 2) Export predictions for complementary fusion
PRED_MAIN_BASE="$EXPORT_ROOT/main_base.json"
PRED_MAIN_TTA="$EXPORT_ROOT/main_tta.json"

python export_gaiic_coco.py \
  --weights "$MAIN_FINAL_WEIGHTS" --split test --imgsz "$MAIN_E6_IMGSZ" \
  --conf "$CONF" --iou "$IOU" --max-det "$MAX_DET" \
  --device "$DEVICE" --batch "$BATCH" \
  --out "$PRED_MAIN_BASE"

if [[ "$EXPORT_TTA" == "1" ]]; then
  python export_gaiic_coco.py \
    --weights "$MAIN_FINAL_WEIGHTS" --split test --imgsz "$MAIN_E6_IMGSZ" \
    --conf "$CONF" --iou "$IOU" --max-det "$MAX_DET" \
    --device "$DEVICE" --batch "$BATCH" --augment \
    --out "$PRED_MAIN_TTA"
fi

PRED_AUX_BASE=""
PRED_AUX_TTA=""
if [[ -n "$AUX_WEIGHTS" ]]; then
  PRED_AUX_BASE="$EXPORT_ROOT/aux_base.json"
  PRED_AUX_TTA="$EXPORT_ROOT/aux_tta.json"
  python export_gaiic_coco.py \
    --weights "$AUX_WEIGHTS" --split test --imgsz "$MAIN_IMGSZ" \
    --conf "$CONF" --iou "$IOU" --max-det "$MAX_DET" \
    --device "$DEVICE" --batch "$BATCH" \
    --out "$PRED_AUX_BASE"

  if [[ "$EXPORT_TTA" == "1" ]]; then
    python export_gaiic_coco.py \
      --weights "$AUX_WEIGHTS" --split test --imgsz "$MAIN_IMGSZ" \
      --conf "$CONF" --iou "$IOU" --max-det "$MAX_DET" \
      --device "$DEVICE" --batch "$BATCH" --augment \
      --out "$PRED_AUX_TTA"
  fi
fi

# 3) Build WBF grid
WBF_MANIFEST="$RUN_ROOT/wbf_manifest.csv"
echo "name,inputs,weights,iou_thr,score_thr,output" > "$WBF_MANIFEST"

IFS=',' read -r -a IOU_LIST <<< "$WBF_IOUS"
IFS=',' read -r -a SCORE_LIST <<< "$WBF_SCORES"

for wiou in "${IOU_LIST[@]}"; do
  for wscore in "${SCORE_LIST[@]}"; do
    inputs=("$PRED_MAIN_BASE")
    weights=("$MAIN_WEIGHT")

    if [[ "$EXPORT_TTA" == "1" && -f "$PRED_MAIN_TTA" ]]; then
      inputs+=("$PRED_MAIN_TTA")
      weights+=("$MAIN_TTA_WEIGHT_SCALE")
    fi

    if [[ -n "$PRED_AUX_BASE" && -f "$PRED_AUX_BASE" ]]; then
      inputs+=("$PRED_AUX_BASE")
      weights+=("$AUX_WEIGHT")
      if [[ "$EXPORT_TTA" == "1" && -f "$PRED_AUX_TTA" ]]; then
        inputs+=("$PRED_AUX_TTA")
        weights+=("$AUX_TTA_WEIGHT_SCALE")
      fi
    fi

    tag="wbf_iou${wiou}_score${wscore}"
    out="$WBF_ROOT/${tag}.json"

    python tools/weighted_box_fusion.py \
      --inputs "${inputs[@]}" \
      --weights "${weights[@]}" \
      --iou-thr "$wiou" --score-thr "$wscore" --max-det "$MAX_DET" \
      --output "$out"

    input_join=$(IFS='|'; echo "${inputs[*]}")
    weight_join=$(IFS='|'; echo "${weights[*]}")
    echo "$tag,$input_join,$weight_join,$wiou,$wscore,$out" >> "$WBF_MANIFEST"
  done
done

# 4) Shrink and collect all candidates
SHRINK_MANIFEST="$RUN_ROOT/submission_manifest.csv"
echo "file,size_mb,pred_records,images_with_pred" > "$SHRINK_MANIFEST"

all_json=()
all_json+=("$PRED_MAIN_BASE")
if [[ "$EXPORT_TTA" == "1" && -f "$PRED_MAIN_TTA" ]]; then
  all_json+=("$PRED_MAIN_TTA")
fi
if [[ -n "$PRED_AUX_BASE" && -f "$PRED_AUX_BASE" ]]; then
  all_json+=("$PRED_AUX_BASE")
fi
if [[ "$EXPORT_TTA" == "1" && -n "$PRED_AUX_TTA" && -f "$PRED_AUX_TTA" ]]; then
  all_json+=("$PRED_AUX_TTA")
fi
while IFS= read -r jf; do
  all_json+=("$jf")
done < <(find "$WBF_ROOT" -maxdepth 1 -type f -name '*.json' | sort)

IFS=',' read -r -a SHRINK_SCORE_LIST <<< "$SHRINK_SCORES"
for src in "${all_json[@]}"; do
  base=$(basename "$src" .json)
  for sthr in "${SHRINK_SCORE_LIST[@]}"; do
    out="$SUB_ROOT/${base}_shrink_s${sthr}.json"
    python tools/shrink_submission.py \
      --input "$src" \
      --output "$out" \
      --score-thr "$sthr" \
      --max-det-per-image "$SHRINK_MAX_DET"
  done
done

python - <<'PY' "$SUB_ROOT" "$SHRINK_MANIFEST"
import csv
import json
import sys
from pathlib import Path

sub_dir = Path(sys.argv[1])
manifest = Path(sys.argv[2])
rows = []
for p in sorted(sub_dir.glob("*.json")):
    data = json.loads(p.read_text(encoding="utf-8"))
    rows.append(
        {
            "file": str(p),
            "size_mb": round(p.stat().st_size / 1024 / 1024, 4),
            "pred_records": len(data),
            "images_with_pred": len({x["image_id"] for x in data}),
        }
    )

with manifest.open("a", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file", "size_mb", "pred_records", "images_with_pred"])
    w.writerows(rows)

print(f"[DONE] submission manifest: {manifest}")
print(f"[DONE] candidates: {len(rows)}")
PY

echo "[DONE] large-model adapted pipeline finished"
echo "[DONE] RUN_ROOT=$RUN_ROOT"
