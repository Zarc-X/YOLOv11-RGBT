#!/usr/bin/env bash
set -euo pipefail

# Fast n-model architecture sweep for midfusion directions.
# MODE:
#   core   -> EFM + EFMv2-P2
#   all    -> EFM + EFMv2-P2 + P5CAS + P5NiN (default)
#   single -> run SINGLE_MODEL only

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/YOLOv11-RGBT}"
DATA_YAML="${DATA_YAML:-ultralytics/cfg/datasets/GAIIC2024-rgbt.yaml}"
PROJECT="${PROJECT:-runs/GAIIC2024_n_arch}"
NAME_PREFIX="${NAME_PREFIX:-n-arch}"
MODE="${MODE:-all}"

EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-2}"
DEVICE="${DEVICE:-0}"
OPTIMIZER="${OPTIMIZER:-SGD}"
CLOSE_MOSAIC="${CLOSE_MOSAIC:-10}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"
DRY_RUN="${DRY_RUN:-0}"

SINGLE_MODEL="${SINGLE_MODEL:-ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion-EFMv2-P2.yaml}"
SINGLE_NAME="${SINGLE_NAME:-n-EFMv2-P2-single}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  set +u
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate yolo11rgbt
  set -u
fi

cd "${PROJECT_ROOT}"

declare -a MODELS
declare -a NAMES

if [[ "${MODE}" == "core" ]]; then
  MODELS=(
    "ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion-EFM.yaml"
    "ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion-EFMv2-P2.yaml"
  )
  NAMES=(
    "n-EFM-baseline"
    "n-EFMv2-P2"
  )
elif [[ "${MODE}" == "all" ]]; then
  MODELS=(
    "ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion-EFM.yaml"
    "ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion-EFMv2-P2.yaml"
    "ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion-EFMv2-P2-P5CAS.yaml"
    "ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion-EFMv2-P2-P5NiN.yaml"
  )
  NAMES=(
    "n-EFM-baseline"
    "n-EFMv2-P2"
    "n-EFMv2-P2-P5CAS"
    "n-EFMv2-P2-P5NiN"
  )
elif [[ "${MODE}" == "single" ]]; then
  MODELS=("${SINGLE_MODEL}")
  NAMES=("${SINGLE_NAME}")
else
  echo "[ERROR] Unsupported MODE=${MODE}. Use: core | all | single"
  exit 1
fi

echo "[INFO] MODE=${MODE}"
echo "[INFO] PROJECT=${PROJECT}"
echo "[INFO] EPOCHS=${EPOCHS} BATCH=${BATCH} IMGSZ=${IMGSZ} DEVICE=${DEVICE}"
echo "[INFO] DRY_RUN=${DRY_RUN}"

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  name="${NAME_PREFIX}-${NAMES[$i]}"

  cmd=(
    python train_RGBT.py
    --model "${model}"
    --data "${DATA_YAML}"
    --epochs "${EPOCHS}"
    --batch "${BATCH}"
    --imgsz "${IMGSZ}"
    --workers "${WORKERS}"
    --device "${DEVICE}"
    --project "${PROJECT}"
    --name "${name}"
    --optimizer "${OPTIMIZER}"
    --close-mosaic "${CLOSE_MOSAIC}"
  )

  if [[ -n "${PRETRAINED_WEIGHTS}" ]]; then
    cmd+=(--pretrained-weights "${PRETRAINED_WEIGHTS}")
  fi

  echo "[RUN] ${name}"
  echo "[CMD] ${cmd[*]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi
  "${cmd[@]}"
done

echo "[DONE] n-model midfusion architecture sweep finished"
