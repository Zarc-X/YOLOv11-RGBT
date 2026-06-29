#!/usr/bin/env bash
set -euo pipefail

# Lightweight high-ROI experiment launcher.
# MODE:
#   all    -> launch configured experiments (sequential by default)
#   status -> print process and log status

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/YOLOv11-RGBT}"
MODE="${MODE:-all}"

BASE_PROJECT="${BASE_PROJECT:-runs/GAIIC2024_lightweight_highroi}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

DATA="${DATA:-ultralytics/cfg/datasets/GAIIC2024-rgbt.yaml}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-2}"
SEED="${SEED:-42}"

# Optional transfer initialization for all experiments.
INIT_WEIGHTS="${INIT_WEIGHTS:-}"

# Two-stage schedule
STAGE1_EPOCHS="${STAGE1_EPOCHS:-30}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-70}"

# Parallel control: default sequential for single-GPU safety.
PARALLEL="${PARALLEL:-0}"

# Augmentation profile
S1_MOSAIC="${S1_MOSAIC:-1.0}"
S2_MOSAIC="${S2_MOSAIC:-0.15}"
S1_MODAL_SHIFT="${S1_MODAL_SHIFT:-0.25}"
S2_MODAL_SHIFT="${S2_MODAL_SHIFT:-0.10}"

cd "$PROJECT_ROOT"

RUN_ROOT="$PROJECT_ROOT/$BASE_PROJECT/$RUN_ID"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$RUN_ROOT" "$LOG_DIR"

if [[ "$MODE" == "status" ]]; then
  echo "[INFO] RUN_ROOT=$RUN_ROOT"
  echo "[INFO] Active lightweight runners:"
  ps -eo pid,etime,cmd | grep -E "run_lightweight_twostage.py|run_lightweight_highroi_parallel.sh" | grep -v grep || true
  echo "[INFO] Recent logs:"
  find "$LOG_DIR" -maxdepth 1 -type f -name "*.log" | sort | while read -r f; do
    echo "----- $f (tail -n 20) -----"
    tail -n 20 "$f" || true
  done
  exit 0
fi

if [[ "$MODE" != "all" ]]; then
  echo "[ERROR] Unsupported MODE=$MODE. Use all|status" >&2
  exit 1
fi

# name|model|imgsz
EXPERIMENTS=(
  "lw-efmv2p2p5nin|ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-EFMv2-P2-P5NiN.yaml|768"
  "lw-efmv2p2p5nin-aux|ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-EFMv2-P2-P5NiN-AUX.yaml|768"
  "lw-deepdbb|ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-DeepDBB.yaml|896"
)

MANIFEST="$RUN_ROOT/launch_manifest.csv"
echo "name,model,imgsz,log" > "$MANIFEST"
LAST_PID=""

launch_one() {
  local exp_name="$1"
  local exp_model="$2"
  local exp_imgsz="$3"
  local log_file="$LOG_DIR/${exp_name}.log"

  local -a cmd=(
    python tools/run_lightweight_twostage.py
    --model "$exp_model"
    --data "$DATA"
    --project "$RUN_ROOT"
    --name "$exp_name"
    --imgsz "$exp_imgsz"
    --batch "$BATCH"
    --workers "$WORKERS"
    --device "$DEVICE"
    --seed "$SEED"
    --stage1-epochs "$STAGE1_EPOCHS"
    --stage2-epochs "$STAGE2_EPOCHS"
    --stage1-mosaic "$S1_MOSAIC"
    --stage2-mosaic "$S2_MOSAIC"
    --stage1-modal-shift "$S1_MODAL_SHIFT"
    --stage2-modal-shift "$S2_MODAL_SHIFT"
  )

  if [[ -n "$INIT_WEIGHTS" ]]; then
    cmd+=(--initial-weights "$INIT_WEIGHTS")
  fi

  echo "[CMD][$exp_name] ${cmd[*]}" >&2
  echo "$exp_name,$exp_model,$exp_imgsz,$log_file" >> "$MANIFEST"

  if [[ "$PARALLEL" == "1" ]]; then
    nohup "${cmd[@]}" > "$log_file" 2>&1 &
    LAST_PID="$!"
    echo "[PID][$exp_name] $LAST_PID" >&2
  else
    "${cmd[@]}" | tee "$log_file"
    LAST_PID=""
  fi
}

pids=()
for item in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r exp_name exp_model exp_imgsz <<< "$item"
  if [[ "$PARALLEL" == "1" ]]; then
    launch_one "$exp_name" "$exp_model" "$exp_imgsz"
    pids+=("$exp_name:$LAST_PID")
  else
    launch_one "$exp_name" "$exp_model" "$exp_imgsz"
  fi
done

if [[ "$PARALLEL" == "1" ]]; then
  for item in "${pids[@]}"; do
    IFS=':' read -r name pid <<< "$item"
    if wait "$pid"; then
      echo "[DONE][$name] pid=$pid"
    else
      echo "[FAIL][$name] pid=$pid"
    fi
  done
fi

echo "[DONE] Launch manifest: $MANIFEST"
echo "[DONE] Run root: $RUN_ROOT"
