#!/usr/bin/env bash
set -euo pipefail

# Big-model upgrade chain:
# 1) train E5/E6 for baseline and big-model branches
# 2) export base/TTA predictions
# 3) run WBF grid for complementary fusion
# 4) run shrink tuning grid and generate manifest

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/GAIIC2024-赛道1-目标检测任务}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-2}"
SEED="${SEED:-42}"

# Baseline branch (current best small/mid model)
BASELINE_WEIGHTS="${BASELINE_WEIGHTS:-runs/GAIIC2024_e1e6_auto/e1e6-r01/e2e4/e2-cont80-img896/weights/best.pt}"
BASELINE_IMGSZ="${BASELINE_IMGSZ:-896}"
BASELINE_E5_BASE_EPOCHS="${BASELINE_E5_BASE_EPOCHS:-15}"
BASELINE_E6_EPOCHS="${BASELINE_E6_EPOCHS:-20}"

# Big-model branch (l/x, 4C or 6C both supported through auto-modal)
BIGMODEL_WEIGHTS="${BIGMODEL_WEIGHTS:-runs/GAIIC2024_mcfx/GAIIC2024-yolo11x-RGBT-MCF-final/weights/best.pt}"
BIGMODEL_IMGSZ="${BIGMODEL_IMGSZ:-896}"
BIGMODEL_E5_BASE_EPOCHS="${BIGMODEL_E5_BASE_EPOCHS:-12}"
BIGMODEL_E6_EPOCHS="${BIGMODEL_E6_EPOCHS:-20}"

# Re-tuning knobs (can override via env)
E5_FOLDS="${E5_FOLDS:-2}"
E5_ROUNDS="${E5_ROUNDS:-1}"
E5_ONLINE_EPOCHS="${E5_ONLINE_EPOCHS:-8}"
E5_OPTIMIZER="${E5_OPTIMIZER:-SGD}"
E5_LR0="${E5_LR0:-0.002}"
E5_LRF="${E5_LRF:-0.01}"
E5_MOMENTUM="${E5_MOMENTUM:-0.937}"
E5_WEIGHT_DECAY="${E5_WEIGHT_DECAY:-0.0005}"

E6_OPTIMIZER="${E6_OPTIMIZER:-SGD}"
E6_LR0="${E6_LR0:-0.0015}"
E6_LRF="${E6_LRF:-0.01}"
E6_MOMENTUM="${E6_MOMENTUM:-0.937}"
E6_WEIGHT_DECAY="${E6_WEIGHT_DECAY:-0.0005}"

# Export/fusion defaults
CONF="${CONF:-0.001}"
IOU="${IOU:-0.6}"
MAX_DET="${MAX_DET:-500}"
EXPORT_TTA="${EXPORT_TTA:-1}"
TTA_WEIGHT_SCALE="${TTA_WEIGHT_SCALE:-0.5}"

WBF_IOUS="${WBF_IOUS:-0.62,0.65,0.68}"
WBF_SCORES="${WBF_SCORES:-0.0005,0.001,0.002}"
WBF_MAX_DET="${WBF_MAX_DET:-500}"
WEIGHT_PAIRS="${WEIGHT_PAIRS:-0.55:0.45,0.50:0.50,0.45:0.55}"

SHRINK_SCORES="${SHRINK_SCORES:-0.008,0.010,0.012}"
SHRINK_MAX_DETS="${SHRINK_MAX_DETS:-100,120,140}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-runs/GAIIC2024_bigmodel_upgrade}"
RUN_ROOT="$OUT_ROOT/$RUN_ID"
TRAIN_DIR="$RUN_ROOT/train"
EXPORT_DIR="$RUN_ROOT/export"
FUSION_DIR="$RUN_ROOT/fusion"
SUB_DIR="$RUN_ROOT/submissions"
LOG_DIR="$RUN_ROOT/logs"

mkdir -p "$TRAIN_DIR" "$EXPORT_DIR" "$FUSION_DIR" "$SUB_DIR" "$LOG_DIR"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  set +u
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate yolo11rgbt || true
  set -u
fi

run_train_chain() {
  local role="$1"
  local weights="$2"
  local imgsz="$3"
  local e5_base_epochs="$4"
  local e6_epochs="$5"

  local project="$TRAIN_DIR/$role"
  local name="${role}-e5e6"
  local log_file="$LOG_DIR/${role}_train.log"

  echo "[INFO][$role] start training chain" | tee -a "$log_file" >&2
  python tools/run_e5_e6_from_weight.py \
    --dataset-root "$DATASET_ROOT" \
    --initial-weights "$weights" \
    --imgsz "$imgsz" \
    --device "$DEVICE" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --e5-folds "$E5_FOLDS" \
    --e5-rounds "$E5_ROUNDS" \
    --e5-base-epochs "$e5_base_epochs" \
    --e5-online-epochs "$E5_ONLINE_EPOCHS" \
    --e5-optimizer "$E5_OPTIMIZER" \
    --e5-lr0 "$E5_LR0" \
    --e5-lrf "$E5_LRF" \
    --e5-momentum "$E5_MOMENTUM" \
    --e5-weight-decay "$E5_WEIGHT_DECAY" \
    --run-e6 \
    --e6-epochs "$e6_epochs" \
    --e6-imgsz "$imgsz" \
    --e6-optimizer "$E6_OPTIMIZER" \
    --e6-lr0 "$E6_LR0" \
    --e6-lrf "$E6_LRF" \
    --e6-momentum "$E6_MOMENTUM" \
    --e6-weight-decay "$E6_WEIGHT_DECAY" \
    --project "$project" \
    --name "$name" \
    --strict-run-root \
    2>&1 | tee -a "$log_file"

  local summary="$project/$name/pipeline_summary.json"
  if [[ ! -f "$summary" ]]; then
      echo "[ERROR][$role] missing summary: $summary" | tee -a "$log_file" >&2
    exit 1
  fi

  local final_w
  final_w=$(python - <<PY
import json
from pathlib import Path
p = Path("$summary")
obj = json.loads(p.read_text(encoding="utf-8"))
e6 = obj.get("e6", {})
print(e6.get("weights", ""))
PY
)
  if [[ -z "$final_w" || ! -f "$final_w" ]]; then
      echo "[ERROR][$role] invalid final weight from summary: $final_w" | tee -a "$log_file" >&2
    exit 1
  fi

  echo "$final_w"
}

export_one_model() {
  local role="$1"
  local weights="$2"
  local imgsz="$3"
  local base_json="$EXPORT_DIR/${role}_base.json"
  local tta_json="$EXPORT_DIR/${role}_tta.json"
  local log_file="$LOG_DIR/${role}_export.log"

  python export_gaiic_coco.py \
    --weights "$weights" --split test --imgsz "$imgsz" \
    --conf "$CONF" --iou "$IOU" --max-det "$MAX_DET" \
    --device "$DEVICE" --batch "$BATCH" \
    --out "$base_json" \
    2>&1 | tee -a "$log_file"

  if [[ "$EXPORT_TTA" == "1" ]]; then
    python export_gaiic_coco.py \
      --weights "$weights" --split test --imgsz "$imgsz" \
      --conf "$CONF" --iou "$IOU" --max-det "$MAX_DET" \
      --device "$DEVICE" --batch "$BATCH" --augment \
      --out "$tta_json" \
      2>&1 | tee -a "$log_file"
  fi
}

csv_split() {
  local raw="$1"
  local -n out_arr="$2"
  out_arr=()
  local IFS=','
  read -r -a out_arr <<< "$raw"
}

build_manifest() {
  python - <<PY
import csv
import json
from pathlib import Path

sub_dir = Path("$SUB_DIR")
manifest = sub_dir / "submission_manifest.csv"
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
with manifest.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "size_mb", "pred_records", "images_with_pred"])
    writer.writeheader()
    writer.writerows(rows)
print(f"manifest {manifest}")
for r in rows:
    print(r)
PY
}

baseline_final=$(run_train_chain "baseline" "$BASELINE_WEIGHTS" "$BASELINE_IMGSZ" "$BASELINE_E5_BASE_EPOCHS" "$BASELINE_E6_EPOCHS")
bigmodel_final=$(run_train_chain "bigmodel" "$BIGMODEL_WEIGHTS" "$BIGMODEL_IMGSZ" "$BIGMODEL_E5_BASE_EPOCHS" "$BIGMODEL_E6_EPOCHS")

echo "[INFO] baseline_final=$baseline_final"
echo "[INFO] bigmodel_final=$bigmodel_final"

export_one_model "baseline" "$baseline_final" "$BASELINE_IMGSZ"
export_one_model "bigmodel" "$bigmodel_final" "$BIGMODEL_IMGSZ"

inputs=("$EXPORT_DIR/baseline_base.json" "$EXPORT_DIR/bigmodel_base.json")

wbf_ious=()
wbf_scores=()
weight_pairs=()
shrink_scores=()
shrink_max_dets=()

csv_split "$WBF_IOUS" wbf_ious
csv_split "$WBF_SCORES" wbf_scores
csv_split "$WEIGHT_PAIRS" weight_pairs
csv_split "$SHRINK_SCORES" shrink_scores
csv_split "$SHRINK_MAX_DETS" shrink_max_dets

if [[ "$EXPORT_TTA" == "1" ]]; then
  inputs+=("$EXPORT_DIR/bigmodel_tta.json")
fi

for pair in "${weight_pairs[@]}"; do
  IFS=':' read -r w_base w_big <<< "$pair"
  if [[ -z "$w_base" || -z "$w_big" ]]; then
    echo "[WARN] skip invalid WEIGHT_PAIRS item: $pair"
    continue
  fi

  if [[ "$EXPORT_TTA" == "1" ]]; then
    w_tta=$(python - <<PY
wb = float("$w_big")
scale = float("$TTA_WEIGHT_SCALE")
print(wb * scale)
PY
)
    weights=("$w_base" "$w_big" "$w_tta")
  else
    weights=("$w_base" "$w_big")
  fi

  for iou_thr in "${wbf_ious[@]}"; do
    for score_thr in "${wbf_scores[@]}"; do
      tag="wb${w_base}_x${w_big}_iou${iou_thr}_s${score_thr}"
      tag=${tag//./p}
      fused="$FUSION_DIR/wbf_${tag}.json"

      python tools/weighted_box_fusion.py \
        --inputs "${inputs[@]}" \
        --weights "${weights[@]}" \
        --iou-thr "$iou_thr" --score-thr "$score_thr" --max-det "$WBF_MAX_DET" \
        --output "$fused"

      for s_thr in "${shrink_scores[@]}"; do
        for s_max in "${shrink_max_dets[@]}"; do
          out="$SUB_DIR/${tag}_shr${s_thr}_m${s_max}.json"
          python tools/shrink_submission.py \
            --input "$fused" \
            --output "$out" \
            --score-thr "$s_thr" \
            --max-det-per-image "$s_max"
        done
      done
    done
  done
done

for src in "$EXPORT_DIR"/*.json; do
  b=$(basename "$src" .json)
  python tools/shrink_submission.py \
    --input "$src" \
    --output "$SUB_DIR/${b}_shrink.json" \
    --score-thr "$(echo "$SHRINK_SCORES" | cut -d',' -f1)" \
    --max-det-per-image "$(echo "$SHRINK_MAX_DETS" | cut -d',' -f1)"
done

build_manifest

echo "[DONE] bigmodel upgrade chain outputs: $RUN_ROOT"
