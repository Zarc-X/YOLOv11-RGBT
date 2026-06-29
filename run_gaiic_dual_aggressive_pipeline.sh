#!/usr/bin/env bash
set -euo pipefail

# One-shot dual pipeline launcher.
# Stage A: category-balanced + online hard-negative + k-fold selection.
# Stage B: x-MCF mainline pipeline.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/YOLOv11-RGBT}"
MODE="${MODE:-both}" # both | balanced | mcf

# --------------------------- Stage A config ---------------------------
BALANCED_DATASET_ROOT="${BALANCED_DATASET_ROOT:-/root/autodl-tmp/GAIIC2024-赛道1-目标检测任务}"
BALANCED_TRAIN_JSON="${BALANCED_TRAIN_JSON:-}"
BALANCED_INITIAL_WEIGHTS="${BALANCED_INITIAL_WEIGHTS:-runs/GAIIC2024_stage2/stage2-midfusion-hardneg3/weights/best.pt}"

BALANCED_PROJECT="${BALANCED_PROJECT:-runs/GAIIC2024_aggressive}"
BALANCED_NAME="${BALANCED_NAME:-balanced-hardneg-kfold}"
BALANCED_TIMESTAMP="${BALANCED_TIMESTAMP:-1}"
BALANCED_DRY_RUN="${BALANCED_DRY_RUN:-0}"

BALANCED_FOLDS="${BALANCED_FOLDS:-3}"
BALANCED_ROUNDS="${BALANCED_ROUNDS:-2}"
BALANCED_BASE_EPOCHS="${BALANCED_BASE_EPOCHS:-20}"
BALANCED_ONLINE_EPOCHS="${BALANCED_ONLINE_EPOCHS:-10}"

BALANCED_IMG_SIZE="${BALANCED_IMG_SIZE:-640}"
BALANCED_BATCH="${BALANCED_BATCH:-16}"
BALANCED_WORKERS="${BALANCED_WORKERS:-2}"
BALANCED_DEVICE="${BALANCED_DEVICE:-0}"
BALANCED_SEED="${BALANCED_SEED:-42}"

BALANCED_MAX_REPEAT="${BALANCED_MAX_REPEAT:-4}"
BALANCED_BALANCE_STRENGTH="${BALANCED_BALANCE_STRENGTH:-0.45}"
BALANCED_HARDNEG_SCORE_THR="${BALANCED_HARDNEG_SCORE_THR:-0.25}"
BALANCED_HARDNEG_IOU_BG_THR="${BALANCED_HARDNEG_IOU_BG_THR:-0.1}"
BALANCED_HARDNEG_MAX_SAMPLES="${BALANCED_HARDNEG_MAX_SAMPLES:-3000}"
BALANCED_HARDNEG_REPEAT="${BALANCED_HARDNEG_REPEAT:-2}"

BALANCED_USE_SIMOTM="${BALANCED_USE_SIMOTM:-RGBT}"
BALANCED_CHANNELS="${BALANCED_CHANNELS:-4}"
BALANCED_PAIRS_RGB="${BALANCED_PAIRS_RGB:-rgb}"
BALANCED_PAIRS_IR="${BALANCED_PAIRS_IR:-tir}"

# --------------------------- Stage B config ---------------------------
MCF_DATA_YAML="${MCF_DATA_YAML:-ultralytics/cfg/datasets/GAIIC2024-rgbt.yaml}"
MCF_DEVICE="${MCF_DEVICE:-0}"
MCF_WORKERS="${MCF_WORKERS:-2}"
MCF_BATCH="${MCF_BATCH:-16}"
MCF_IMGSZ="${MCF_IMGSZ:-640}"

MCF_EPOCHS_STEP1="${MCF_EPOCHS_STEP1:-100}"
MCF_EPOCHS_STEP2="${MCF_EPOCHS_STEP2:-1}"
MCF_FRACTION_STEP2="${MCF_FRACTION_STEP2:-0.01}"
MCF_EPOCHS_STEP4="${MCF_EPOCHS_STEP4:-100}"

MCF_RUN_PROJECT="${MCF_RUN_PROJECT:-runs/GAIIC2024_mcfx}"
MCF_STEP1_NAME="${MCF_STEP1_NAME:-GAIIC2024-yolo11x-RGB-main}"
MCF_STEP2_NAME="${MCF_STEP2_NAME:-GAIIC2024-yolo11x-RGBT-MCF-template}"
MCF_STEP4_NAME="${MCF_STEP4_NAME:-GAIIC2024-yolo11x-RGBT-MCF-final}"

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] MODE=${MODE}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  set +u
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate yolo11rgbt
  set -u
fi

cd "${PROJECT_ROOT}"

if [[ "${MODE}" == "both" || "${MODE}" == "balanced" ]]; then
  echo "[STAGE A] category-balanced + online hard-negative + k-fold selection"
  balanced_cmd=(
    python tools/run_balanced_hardneg_kfold_pipeline.py
    --dataset-root "${BALANCED_DATASET_ROOT}"
    --initial-weights "${BALANCED_INITIAL_WEIGHTS}"
    --project "${BALANCED_PROJECT}"
    --name "${BALANCED_NAME}"
    --folds "${BALANCED_FOLDS}"
    --rounds "${BALANCED_ROUNDS}"
    --base-epochs "${BALANCED_BASE_EPOCHS}"
    --online-epochs "${BALANCED_ONLINE_EPOCHS}"
    --imgsz "${BALANCED_IMG_SIZE}"
    --batch "${BALANCED_BATCH}"
    --workers "${BALANCED_WORKERS}"
    --device "${BALANCED_DEVICE}"
    --seed "${BALANCED_SEED}"
    --balance-max-repeat "${BALANCED_MAX_REPEAT}"
    --balance-strength "${BALANCED_BALANCE_STRENGTH}"
    --hardneg-score-thr "${BALANCED_HARDNEG_SCORE_THR}"
    --hardneg-iou-bg-thr "${BALANCED_HARDNEG_IOU_BG_THR}"
    --hardneg-max-samples "${BALANCED_HARDNEG_MAX_SAMPLES}"
    --hardneg-repeat "${BALANCED_HARDNEG_REPEAT}"
    --use-simotm "${BALANCED_USE_SIMOTM}"
    --channels "${BALANCED_CHANNELS}"
    --pairs-rgb "${BALANCED_PAIRS_RGB}"
    --pairs-ir "${BALANCED_PAIRS_IR}"
  )

  if [[ -n "${BALANCED_TRAIN_JSON}" ]]; then
    balanced_cmd+=(--train-json "${BALANCED_TRAIN_JSON}")
  fi
  if [[ "${BALANCED_TIMESTAMP}" == "1" ]]; then
    balanced_cmd+=(--timestamp)
  fi
  if [[ "${BALANCED_DRY_RUN}" == "1" ]]; then
    balanced_cmd+=(--dry-run)
  fi

  echo "[CMD] ${balanced_cmd[*]}"
  "${balanced_cmd[@]}"
fi

if [[ "${MODE}" == "both" || "${MODE}" == "mcf" ]]; then
  echo "[STAGE B] x-MCF mainline"
  PROJECT_ROOT="${PROJECT_ROOT}" \
  DATA_YAML="${MCF_DATA_YAML}" \
  DEVICE="${MCF_DEVICE}" \
  WORKERS="${MCF_WORKERS}" \
  BATCH="${MCF_BATCH}" \
  IMGSZ="${MCF_IMGSZ}" \
  EPOCHS_STEP1="${MCF_EPOCHS_STEP1}" \
  EPOCHS_STEP2="${MCF_EPOCHS_STEP2}" \
  FRACTION_STEP2="${MCF_FRACTION_STEP2}" \
  EPOCHS_STEP4="${MCF_EPOCHS_STEP4}" \
  RUN_PROJECT="${MCF_RUN_PROJECT}" \
  STEP1_NAME="${MCF_STEP1_NAME}" \
  STEP2_NAME="${MCF_STEP2_NAME}" \
  STEP4_NAME="${MCF_STEP4_NAME}" \
  bash run_gaiic_mcf_x_pipeline.sh
fi

echo "[DONE] dual aggressive pipeline complete"
