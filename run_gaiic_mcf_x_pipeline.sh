#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline for paper-style strongest MCF training flow on GAIIC.
# Steps:
#   1) Train single-modal RGB main branch (YOLO11x)
#   2) Train MCF template for 1 epoch with fraction=0.01
#   3) Convert weights with transform_MCF.py
#   4) Final MCF training from converted checkpoint

# ---------------------- user-tunable defaults ----------------------
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/YOLOv11-RGBT}"
DATA_YAML="${DATA_YAML:-ultralytics/cfg/datasets/GAIIC2024-rgbt.yaml}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-2}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"

EPOCHS_STEP1="${EPOCHS_STEP1:-100}"
EPOCHS_STEP2="${EPOCHS_STEP2:-1}"
FRACTION_STEP2="${FRACTION_STEP2:-0.01}"
EPOCHS_STEP4="${EPOCHS_STEP4:-100}"

RUN_PROJECT="${RUN_PROJECT:-runs/GAIIC2024}"
STEP1_NAME="${STEP1_NAME:-GAIIC2024-yolo11x-RGB-main}"
STEP2_NAME="${STEP2_NAME:-GAIIC2024-yolo11x-RGBT-MCF-template}"
STEP4_NAME="${STEP4_NAME:-GAIIC2024-yolo11x-RGBT-MCF-final}"

STEP1_MODEL_YAML="${STEP1_MODEL_YAML:-ultralytics/cfg/models/11/yolo11x.yaml}"
MCF_YAML_TEMPLATE="${MCF_YAML_TEMPLATE:-ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-MCF.yaml}"
MCF_YAML_X="${MCF_YAML_X:-ultralytics/cfg/models/11-RGBT/yolo11x-RGBT-midfusion-MCF.yaml}"

STEP1_BEST="${STEP1_BEST:-${RUN_PROJECT}/${STEP1_NAME}/weights/best.pt}"
STEP1_LAST="${STEP1_LAST:-${RUN_PROJECT}/${STEP1_NAME}/weights/last.pt}"
STEP2_BEST="${STEP2_BEST:-${RUN_PROJECT}/${STEP2_NAME}/weights/best.pt}"
STEP2_LAST="${STEP2_LAST:-${RUN_PROJECT}/${STEP2_NAME}/weights/last.pt}"
MCF_INIT_PT="${MCF_INIT_PT:-${RUN_PROJECT}/GAIIC2024-yolo11x-RGBT-MCF.pt}"
STEP4_BEST="${STEP4_BEST:-${RUN_PROJECT}/${STEP4_NAME}/weights/best.pt}"
STEP4_LAST="${STEP4_LAST:-${RUN_PROJECT}/${STEP4_NAME}/weights/last.pt}"

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] DATA_YAML=${DATA_YAML}"
echo "[INFO] RUN_PROJECT=${RUN_PROJECT}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # Activate the validated environment automatically.
  set +u
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate yolo11rgbt
  set -u
fi

cd "${PROJECT_ROOT}"

if [[ ! -f "${DATA_YAML}" ]]; then
  echo "[ERROR] DATA_YAML not found: ${DATA_YAML}" >&2
  exit 1
fi

if [[ ! -f "${MCF_YAML_X}" ]]; then
  if [[ ! -f "${MCF_YAML_TEMPLATE}" ]]; then
    echo "[ERROR] MCF template yaml not found: ${MCF_YAML_TEMPLATE}" >&2
    exit 1
  fi
  cp "${MCF_YAML_TEMPLATE}" "${MCF_YAML_X}"
  echo "[INFO] Created ${MCF_YAML_X} from template."
fi

mkdir -p "$(dirname "${MCF_INIT_PT}")"

echo "[STEP1] Train RGB main branch (or resume)..."
if [[ -f "${STEP1_BEST}" ]]; then
  echo "[STEP1] best.pt exists, skip: ${STEP1_BEST}"
else
  python - <<PY
from pathlib import Path
from ultralytics import YOLO

step1_model_yaml = "${STEP1_MODEL_YAML}"
step1_last = Path("${STEP1_LAST}")
data_yaml = "${DATA_YAML}"

if step1_last.exists():
    model = YOLO(str(step1_last))
    model.train(resume=True)
else:
    model = YOLO(step1_model_yaml)
    model.load("yolo11x.pt")
    model.train(
        data=data_yaml,
        cache=False,
        imgsz=${IMGSZ},
        epochs=${EPOCHS_STEP1},
        batch=${BATCH},
        close_mosaic=0,
        workers=${WORKERS},
        device="${DEVICE}",
        optimizer="SGD",
        use_simotm="RGB",
        channels=3,
        project="${RUN_PROJECT}",
        name="${STEP1_NAME}",
        resume=False,
    )
PY
fi

echo "[STEP2] Train MCF template (or resume)..."
if [[ -f "${STEP2_BEST}" ]]; then
  echo "[STEP2] best.pt exists, skip: ${STEP2_BEST}"
else
  python - <<PY
from pathlib import Path
from ultralytics import YOLO

mcf_yaml_x = "${MCF_YAML_X}"
step2_last = Path("${STEP2_LAST}")
data_yaml = "${DATA_YAML}"

if step2_last.exists():
    model = YOLO(str(step2_last))
    model.train(resume=True)
else:
    model = YOLO(mcf_yaml_x)
    model.train(
        data=data_yaml,
        cache=False,
        imgsz=${IMGSZ},
        epochs=${EPOCHS_STEP2},
        fraction=${FRACTION_STEP2},
        batch=${BATCH},
        close_mosaic=0,
        workers=${WORKERS},
        device="${DEVICE}",
        optimizer="SGD",
        use_simotm="RGBRGB6C",
        channels=6,
        pairs_rgb_ir=["rgb", "tir"],
        project="${RUN_PROJECT}",
        name="${STEP2_NAME}",
        resume=False,
    )
PY
fi

echo "[STEP3] Convert Step1/Step2 weights -> MCF init checkpoint..."
if [[ -f "${MCF_INIT_PT}" && "${STEP1_BEST}" -ot "${MCF_INIT_PT}" && "${STEP2_BEST}" -ot "${MCF_INIT_PT}" ]]; then
  echo "[STEP3] Converted checkpoint up-to-date, skip: ${MCF_INIT_PT}"
else
  python transform_MCF.py \
    --source-model-path "${STEP1_BEST}" \
    --target-model-path "${STEP2_BEST}" \
    --output-model-path "${MCF_INIT_PT}"
fi

echo "[STEP4] Final MCF training (or resume)..."
if [[ -f "${STEP4_BEST}" ]]; then
  echo "[STEP4] best.pt exists, skip: ${STEP4_BEST}"
else
  python - <<PY
from pathlib import Path
from ultralytics import YOLO

step4_last = Path("${STEP4_LAST}")
mcf_init_pt = Path("${MCF_INIT_PT}")
if not mcf_init_pt.exists():
    raise FileNotFoundError(f"MCF init checkpoint not found: {mcf_init_pt}")

if step4_last.exists():
    model = YOLO(str(step4_last))
    model.train(resume=True)
else:
    model = YOLO(str(mcf_init_pt))
    model.train(
        data="${DATA_YAML}",
        cache=False,
        imgsz=${IMGSZ},
        epochs=${EPOCHS_STEP4},
        batch=${BATCH},
        close_mosaic=0,
        workers=${WORKERS},
        device="${DEVICE}",
        optimizer="Adam",
        lr0=0.001,
        warmup_epochs=1.0,
        warmup_bias_lr=0.01,
        freeze=[2, 3, 4, 5, 6, 17, 18, 23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
        use_simotm="RGBRGB6C",
        channels=6,
        pairs_rgb_ir=["rgb", "tir"],
        project="${RUN_PROJECT}",
        name="${STEP4_NAME}",
        resume=False,
    )
PY
fi

echo "[DONE] Pipeline finished."
echo "[DONE] Step1 best: ${STEP1_BEST}"
echo "[DONE] Step2 best: ${STEP2_BEST}"
echo "[DONE] MCF init:  ${MCF_INIT_PT}"
echo "[DONE] Step4 best: ${STEP4_BEST}"
