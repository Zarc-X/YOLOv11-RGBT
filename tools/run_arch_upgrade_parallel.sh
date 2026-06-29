#!/usr/bin/env bash
set -euo pipefail

# Parallel architecture-upgrade pipeline
# MODE:
#   all    -> phase1 (E2/E3/E4 parallel gate) + phase2 (TopK E5/E6 parallel)
#   phase1 -> only run E2/E3/E4 gate in parallel
#   phase2 -> only run TopK E5/E6 in parallel (requires phase1 summaries)
#   status -> print run status and recent logs

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/YOLOv11-RGBT}"
MODE="${MODE:-all}"

BASE_PROJECT="${BASE_PROJECT:-runs/GAIIC2024_arch_upgrade_parallel}"
RUN_ID="${RUN_ID:-}"

DATA="${DATA:-ultralytics/cfg/datasets/GAIIC2024-rgbt.yaml}"
STAGE2_DATA="${STAGE2_DATA:-runs/analysis/stage2_hardneg_midfusion/GAIIC2024-rgbt-stage2-hardneg.yaml}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-2}"
SEED="${SEED:-42}"

# Strong default seed from historical best branch.
SEED_WEIGHTS="${SEED_WEIGHTS:-runs/GAIIC2024_mcfx/GAIIC2024-yolo11x-RGBT-MCF-final/weights/best.pt}"

# Stop-loss settings for phase1.
ANCHOR_MAP="${ANCHOR_MAP:-0.620}"
EARLY_DROP_GAP="${EARLY_DROP_GAP:-0.003}"
MIN_GAIN="${MIN_GAIN:-0.002}"
PILOT_EPOCHS="${PILOT_EPOCHS:-20}"
FULL_EPOCHS="${FULL_EPOCHS:-100}"
PHASE1_DISABLE_RULE1="${PHASE1_DISABLE_RULE1:-0}"
PHASE1_DISABLE_RULE2="${PHASE1_DISABLE_RULE2:-1}"
ALLOW_STOPPED_BY_RULE1="${ALLOW_STOPPED_BY_RULE1:-0}"

# Parallel branch architecture plan.
E2_MODEL="${E2_MODEL:-ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-DeepDBB.yaml}"
E3_MODEL="${E3_MODEL:-ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-EFM.yaml}"
E4_MODEL="${E4_MODEL:-ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-NiNfusion.yaml}"
E2_IMGSZ="${E2_IMGSZ:-896}"
E3_IMGSZ="${E3_IMGSZ:-768}"
E4_IMGSZ="${E4_IMGSZ:-896}"

# Phase2 settings.
TOPK="${TOPK:-2}"
E5_FOLDS="${E5_FOLDS:-3}"
E5_ROUNDS="${E5_ROUNDS:-1}"
E5_BASE_EPOCHS="${E5_BASE_EPOCHS:-20}"
E5_ONLINE_EPOCHS="${E5_ONLINE_EPOCHS:-10}"
RUN_E6="${RUN_E6:-1}"
E6_EPOCHS="${E6_EPOCHS:-30}"
E6_IMGSZ="${E6_IMGSZ:-768}" # use "match" to follow each branch imgsz
FORCE_RELAUNCH="${FORCE_RELAUNCH:-0}"

latest_run_id() {
  local base_dir="$1"
  if [[ ! -d "$base_dir" ]]; then
    return 1
  fi
  ls -1dt "$base_dir"/* 2>/dev/null | head -n 1 | xargs -r basename
}

ensure_run_id() {
  local base_abs="$1"
  case "$MODE" in
    all|phase1)
      if [[ -z "$RUN_ID" ]]; then
        RUN_ID="$(date +%Y%m%d-%H%M%S)"
      fi
      ;;
    phase2|status)
      if [[ -z "$RUN_ID" ]]; then
        RUN_ID="$(latest_run_id "$base_abs" || true)"
      fi
      if [[ -z "$RUN_ID" ]]; then
        echo "[ERROR] RUN_ID is empty and no previous run found under $base_abs" >&2
        exit 1
      fi
      ;;
    *)
      echo "[ERROR] Unsupported MODE=$MODE. Use: all | phase1 | phase2 | status" >&2
      exit 1
      ;;
  esac
}

launch_phase1_branch() {
  local tag="$1"
  local name="$2"
  local log_file="$3"
  local pid_var="$4"

  local -a cmd=(
    python tools/run_e1_to_e6_pipeline.py
    --skip-e1
    --anchor-map "$ANCHOR_MAP"
    --branch-only
    --run-tags "$tag"
    --pilot-epochs "$PILOT_EPOCHS"
    --full-epochs "$FULL_EPOCHS"
    --early-drop-gap "$EARLY_DROP_GAP"
    --min-gain "$MIN_GAIN"
    --seed-weights "$SEED_WEIGHTS"
    --e2-model "$E2_MODEL"
    --e3-model "$E3_MODEL"
    --e4-model "$E4_MODEL"
    --e2-imgsz "$E2_IMGSZ"
    --e3-imgsz "$E3_IMGSZ"
    --e4-imgsz "$E4_IMGSZ"
    --data "$DATA"
    --stage2-data "$STAGE2_DATA"
    --device "$DEVICE"
    --batch "$BATCH"
    --workers "$WORKERS"
    --seed "$SEED"
    --project "$PHASE1_PROJECT"
    --name "$name"
  )

  if [[ "$PHASE1_DISABLE_RULE1" == "1" ]]; then
    cmd+=(--disable-rule1)
  fi
  if [[ "$PHASE1_DISABLE_RULE2" == "1" ]]; then
    cmd+=(--disable-rule2)
  fi

  # Keep stdout clean for PID-only return when used in command substitution.
  echo "[CMD][$tag] ${cmd[*]}" >&2
  nohup "${cmd[@]}" >"$log_file" 2>&1 &
  local pid="$!"
  printf -v "$pid_var" '%s' "$pid"
  echo "[PID][$tag] $pid" >&2
}

build_phase1_leaderboard() {
  local e2_summary="$1"
  local e3_summary="$2"
  local e4_summary="$3"
  local out_csv="$4"
  local out_topk="$5"

  ALLOW_STOPPED_BY_RULE1="$ALLOW_STOPPED_BY_RULE1" TOPK="$TOPK" python - "$e2_summary" "$e3_summary" "$e4_summary" "$out_csv" "$out_topk" <<'PY'
import csv
import json
import math
import os
import sys
from pathlib import Path

summary_paths = {
    "E2": Path(sys.argv[1]),
    "E3": Path(sys.argv[2]),
    "E4": Path(sys.argv[3]),
}
out_csv = Path(sys.argv[4])
out_topk = Path(sys.argv[5])
allow_stopped = os.environ.get("ALLOW_STOPPED_BY_RULE1", "0") == "1"
topk = int(os.environ.get("TOPK", "2"))

rows = []
for tag, p in summary_paths.items():
    row = {
        "tag": tag,
        "summary_path": str(p),
        "status": "missing",
        "best_map50_95": float("nan"),
        "best_weights": "",
        "imgsz": "",
    }
    if p.exists():
        obj = json.loads(p.read_text(encoding="utf-8"))
        branch = None
        for rec in obj.get("e2_e4", []):
            if str(rec.get("tag", "")).upper() == tag:
                branch = rec
                break
        if branch is not None:
            row["status"] = str(branch.get("status", "unknown"))
            try:
                row["best_map50_95"] = float(branch.get("best_map50_95", float("nan")))
            except Exception:
                row["best_map50_95"] = float("nan")
            row["best_weights"] = str(branch.get("best_weights", ""))
            row["imgsz"] = str(branch.get("imgsz", ""))
    rows.append(row)

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["tag", "status", "best_map50_95", "imgsz", "best_weights", "summary_path"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

def eligible(r):
    st = r["status"]
    if st in {"missing", "skipped", "skipped_by_selection"}:
        return False
    if (not allow_stopped) and st == "stopped_by_rule1":
        return False
    if not r["best_weights"]:
        return False
    v = r["best_map50_95"]
    return isinstance(v, float) and math.isfinite(v)

usable = [r for r in rows if eligible(r)]
usable.sort(key=lambda x: x["best_map50_95"], reverse=True)
top = usable[: max(1, topk)]

out_topk.write_text(json.dumps(top, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[INFO] phase1 leaderboard -> {out_csv}")
print(f"[INFO] phase1 topk -> {out_topk}")
for i, r in enumerate(top, start=1):
    print(f"[TOP{i}] {r['tag']} map={r['best_map50_95']:.6f} imgsz={r['imgsz']} status={r['status']}")
PY
}

launch_phase2_topk() {
  local topk_json="$1"
  local launch_manifest="$2"

  mapfile -t items < <(python - "$topk_json" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(0)
obj = json.loads(p.read_text(encoding="utf-8"))
for r in obj:
    print("|".join([
        str(r.get("tag", "")),
        str(r.get("imgsz", "")),
        str(r.get("best_map50_95", "")),
        str(r.get("best_weights", "")),
    ]))
PY
  )

  if [[ "${#items[@]}" -eq 0 ]]; then
    echo "[ERROR] No usable TopK items found in $topk_json" >&2
    exit 1
  fi

  : >"$launch_manifest"
  local pids=()
  for line in "${items[@]}"; do
    IFS='|' read -r tag imgsz best_map best_weights <<<"$line"
    if [[ -z "$tag" || -z "$best_weights" ]]; then
      continue
    fi

    local e6_imgsz="$E6_IMGSZ"
    if [[ "$E6_IMGSZ" == "match" ]]; then
      e6_imgsz="$imgsz"
    fi

    local name="phase2-${tag,,}-${RUN_ID}"
    local log_file="$LOG_DIR/${name}.log"

    local -a cmd=(
      python tools/run_e5_e6_from_weight.py
      --initial-weights "$best_weights"
      --imgsz "$imgsz"
      --device "$DEVICE"
      --batch "$BATCH"
      --workers "$WORKERS"
      --seed "$SEED"
      --e5-folds "$E5_FOLDS"
      --e5-rounds "$E5_ROUNDS"
      --e5-base-epochs "$E5_BASE_EPOCHS"
      --e5-online-epochs "$E5_ONLINE_EPOCHS"
      --stage2-data "$STAGE2_DATA"
      --e6-epochs "$E6_EPOCHS"
      --e6-imgsz "$e6_imgsz"
      --project "$PHASE2_PROJECT"
      --name "$name"
    )

    if [[ "$RUN_E6" == "1" ]]; then
      cmd+=(--run-e6)
    fi

    echo "[CMD][${tag}] ${cmd[*]}"
    nohup "${cmd[@]}" >"$log_file" 2>&1 &
    local pid="$!"
    pids+=("${tag}:${pid}")
    echo "${tag},${pid},${name},${best_map},${imgsz},${best_weights},${log_file}" >>"$launch_manifest"
  done

  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "[ERROR] No phase2 job launched." >&2
    exit 1
  fi

  echo "[INFO] phase2 launched manifest -> $launch_manifest"
  for item in "${pids[@]}"; do
    IFS=':' read -r tag pid <<<"$item"
    if wait "$pid"; then
      echo "[DONE][${tag}] pid=${pid}"
    else
      echo "[FAIL][${tag}] pid=${pid}"
    fi
  done
}

print_status() {
  echo "[INFO] RUN_ROOT=$RUN_ROOT"
  echo "[INFO] MODE=status"
  echo "[INFO] running processes:"
  ps -eo pid,etime,cmd | grep -E "run_e1_to_e6_pipeline.py|run_e5_e6_from_weight.py" | grep -v grep || true

  if [[ -d "$LOG_DIR" ]]; then
    echo "[INFO] recent logs under $LOG_DIR"
    find "$LOG_DIR" -maxdepth 1 -type f -name "*.log" | sort | while read -r f; do
      echo "----- $f (tail -n 20) -----"
      tail -n 20 "$f" || true
    done
  fi
}

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  set +u
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate yolo11rgbt
  set -u
fi

cd "$PROJECT_ROOT"

BASE_PROJECT_ABS="$PROJECT_ROOT/$BASE_PROJECT"
mkdir -p "$BASE_PROJECT_ABS"
ensure_run_id "$BASE_PROJECT_ABS"

RUN_ROOT="$BASE_PROJECT_ABS/$RUN_ID"
PHASE1_PROJECT="$RUN_ROOT/phase1"
PHASE2_PROJECT="$RUN_ROOT/phase2"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$RUN_ROOT" "$PHASE1_PROJECT" "$PHASE2_PROJECT" "$LOG_DIR"

E2_NAME="phase1-e2-${RUN_ID}"
E3_NAME="phase1-e3-${RUN_ID}"
E4_NAME="phase1-e4-${RUN_ID}"

E2_SUMMARY="$PHASE1_PROJECT/$E2_NAME/pipeline_summary.json"
E3_SUMMARY="$PHASE1_PROJECT/$E3_NAME/pipeline_summary.json"
E4_SUMMARY="$PHASE1_PROJECT/$E4_NAME/pipeline_summary.json"

PHASE1_LEADERBOARD="$RUN_ROOT/phase1_leaderboard.csv"
PHASE1_TOPK="$RUN_ROOT/phase1_topk.json"
PHASE2_MANIFEST="$RUN_ROOT/phase2_launch_manifest.csv"

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] RUN_ID=$RUN_ID"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] MODE=$MODE"

if [[ "$MODE" == "status" ]]; then
  print_status
  exit 0
fi

if [[ "$MODE" == "all" || "$MODE" == "phase1" ]]; then
  if [[ "$FORCE_RELAUNCH" != "1" ]]; then
    existing_phase1="$(ps -eo pid,cmd | awk -v p="$PHASE1_PROJECT" 'index($0,"run_e1_to_e6_pipeline.py") && index($0,"--project " p){c++} END{print c+0}')"
    if [[ "$existing_phase1" -gt 0 ]]; then
      echo "[ERROR] Found ${existing_phase1} existing phase1 jobs for RUN_ID=$RUN_ID. Use MODE=status first or set FORCE_RELAUNCH=1." >&2
      exit 1
    fi
  fi

  echo "[PHASE1] Launch E2/E3/E4 in parallel"
  pid_e2=""
  pid_e3=""
  pid_e4=""
  launch_phase1_branch "E2" "$E2_NAME" "$LOG_DIR/$E2_NAME.log" pid_e2
  launch_phase1_branch "E3" "$E3_NAME" "$LOG_DIR/$E3_NAME.log" pid_e3
  launch_phase1_branch "E4" "$E4_NAME" "$LOG_DIR/$E4_NAME.log" pid_e4

  for p in "$pid_e2" "$pid_e3" "$pid_e4"; do
    if [[ ! "$p" =~ ^[0-9]+$ ]]; then
      echo "[ERROR] Invalid phase1 PID captured: '$p'" >&2
      exit 1
    fi
  done

  phase1_fail=0
  for item in "E2:$pid_e2" "E3:$pid_e3" "E4:$pid_e4"; do
    IFS=':' read -r tag pid <<<"$item"
    if wait "$pid"; then
      echo "[DONE][${tag}] pid=${pid}"
    else
      echo "[FAIL][${tag}] pid=${pid}"
      phase1_fail=1
    fi
  done

  if [[ "$phase1_fail" -ne 0 ]]; then
    echo "[ERROR] phase1 has failed jobs. Check logs under $LOG_DIR and rerun MODE=phase1 or MODE=status." >&2
    exit 1
  fi

  build_phase1_leaderboard "$E2_SUMMARY" "$E3_SUMMARY" "$E4_SUMMARY" "$PHASE1_LEADERBOARD" "$PHASE1_TOPK"
fi

if [[ "$MODE" == "all" || "$MODE" == "phase2" ]]; then
  if [[ "$FORCE_RELAUNCH" != "1" ]]; then
    existing_phase2="$(ps -eo pid,cmd | awk -v p="$PHASE2_PROJECT" 'index($0,"run_e5_e6_from_weight.py") && index($0,"--project " p){c++} END{print c+0}')"
    if [[ "$existing_phase2" -gt 0 ]]; then
      echo "[ERROR] Found ${existing_phase2} existing phase2 jobs for RUN_ID=$RUN_ID. Use MODE=status first or set FORCE_RELAUNCH=1." >&2
      exit 1
    fi
  fi

  if [[ ! -f "$PHASE1_TOPK" ]]; then
    echo "[INFO] phase1_topk.json missing, rebuild from summaries"
    build_phase1_leaderboard "$E2_SUMMARY" "$E3_SUMMARY" "$E4_SUMMARY" "$PHASE1_LEADERBOARD" "$PHASE1_TOPK"
  fi
  echo "[PHASE2] Launch Top${TOPK} E5/E6 in parallel"
  launch_phase2_topk "$PHASE1_TOPK" "$PHASE2_MANIFEST"
fi

echo "[DONE] pipeline finished. RUN_ROOT=$RUN_ROOT"
