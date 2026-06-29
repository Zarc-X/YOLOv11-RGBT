#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VALID_BRANCH_TAGS = ("E2", "E3", "E4")


def log(msg: str) -> None:
    print(msg, flush=True)


def is_finite(v: float | None) -> bool:
    return v is not None and math.isfinite(v) and not math.isnan(v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-run E1 to E6 experiment pipeline with stop-loss rules")

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/root/autodl-tmp/GAIIC2024-赛道1-目标检测任务"),
        help="GAIIC dataset root",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("ultralytics/cfg/datasets/GAIIC2024-rgbt.yaml"),
        help="Main train data yaml for E2/E3/E4",
    )
    parser.add_argument(
        "--stage2-data",
        type=Path,
        default=Path("runs/analysis/stage2_hardneg_midfusion/GAIIC2024-rgbt-stage2-hardneg.yaml"),
        help="Stage2 train data yaml for E6",
    )

    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pilot-epochs", type=int, default=20, help="Pilot epochs for early stop gate")
    parser.add_argument("--full-epochs", type=int, default=100, help="Target total epochs for E2/E3/E4")
    parser.add_argument(
        "--early-drop-gap",
        type=float,
        default=0.003,
        help="Rule-1: drop branch if pilot_map < anchor_map - early_drop_gap",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.002,
        help="Rule-2: only enter next stage when gain >= min_gain",
    )

    parser.add_argument("--e2-model", type=Path, default=Path("ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-P3P4.yaml"))
    parser.add_argument("--e3-model", type=Path, default=Path("ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-P3P4.yaml"))
    parser.add_argument("--e4-model", type=Path, default=Path("ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-P3.yaml"))
    parser.add_argument("--e2-imgsz", type=int, default=896)
    parser.add_argument("--e3-imgsz", type=int, default=768)
    parser.add_argument("--e4-imgsz", type=int, default=896)

    parser.add_argument("--e1-imgsz", type=int, default=640)
    parser.add_argument("--anchor-weights", nargs="*", default=None, help="Weights list for E1 official recheck")
    parser.add_argument("--seed-weights", type=Path, default=None, help="Optional init weights for E2/E3/E4")

    parser.add_argument("--e5-folds", type=int, default=3)
    parser.add_argument("--e5-rounds", type=int, default=1)
    parser.add_argument("--e5-base-epochs", type=int, default=20)
    parser.add_argument("--e5-online-epochs", type=int, default=10)

    parser.add_argument("--e6-epochs", type=int, default=30)
    parser.add_argument("--e6-imgsz", type=int, default=640)

    parser.add_argument("--project", type=Path, default=Path("runs/GAIIC2024_e1e6_auto"))
    parser.add_argument("--name", type=str, default="e1e6")
    parser.add_argument("--timestamp", action="store_true", help="Append timestamp to run name")
    parser.add_argument(
        "--strict-run-root",
        action="store_true",
        help="Fail if target run directory already exists (default: auto suffix on conflict)",
    )

    parser.add_argument(
        "--run-tags",
        type=str,
        default="E2,E3,E4",
        help="Comma-separated branch tags to run from E2,E3,E4. Useful for multi-terminal parallel training.",
    )
    parser.add_argument("--branch-only", action="store_true", help="Run E1/E2-E4 only, skip E5/E6")
    parser.add_argument("--skip-e1", action="store_true", help="Skip official recheck step and reuse provided anchor")
    parser.add_argument("--anchor-map", type=float, default=None, help="Manual anchor mAP50-95 when --skip-e1")
    parser.add_argument(
        "--anchor-source-summary",
        type=Path,
        default=None,
        help="Load anchor_map and seed_weights from an existing pipeline_summary.json",
    )
    parser.add_argument("--disable-rule1", action="store_true", help="Disable pilot early-stop rule")
    parser.add_argument("--disable-rule2", action="store_true", help="Disable stage-gating min-gain rule")

    return parser.parse_args()


def resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (ROOT / p).resolve()


def parse_run_tags(raw: str) -> list[str]:
    tags: list[str] = []
    for x in raw.split(","):
        t = x.strip().upper()
        if not t:
            continue
        if t not in VALID_BRANCH_TAGS:
            raise ValueError(f"Invalid tag '{t}'. Supported tags: {', '.join(VALID_BRANCH_TAGS)}")
        if t not in tags:
            tags.append(t)
    if not tags:
        raise ValueError("--run-tags resolved to empty selection")
    return tags


def load_anchor_from_summary(summary_path: Path) -> tuple[float, Path | None]:
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    e1 = obj.get("e1", {}) if isinstance(obj, dict) else {}
    anchor_raw = e1.get("anchor_map50_95", float("nan"))
    try:
        anchor_map = float(anchor_raw)
    except Exception:
        anchor_map = float("nan")

    seed = e1.get("seed_weights", "")
    seed_path = None
    if isinstance(seed, str) and seed:
        p = Path(seed)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if p.exists():
            seed_path = p
    return anchor_map, seed_path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    log("[CMD] " + " ".join(cmd))
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    root_path = str(ROOT)
    env["PYTHONPATH"] = root_path if not py_path else f"{root_path}:{py_path}"
    subprocess.run(cmd, check=True, cwd=str(cwd), env=env)


def get_yolo_cls():
    try:
        from ultralytics import YOLO  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "ultralytics is not installed in current python env. "
            "Activate the training environment first, then rerun."
        ) from e
    return YOLO


def get_ckpt(run_dir: Path) -> Path:
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found under {run_dir}")


def parse_best_map(results_csv: Path) -> tuple[float, int]:
    if not results_csv.exists():
        return float("nan"), -1

    best_map = float("-inf")
    best_epoch = -1
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(float(row.get("epoch", -1)))
            except Exception:
                epoch = -1

            v = None
            for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95(M)"):
                if key in row and row[key] not in (None, ""):
                    try:
                        v = float(row[key])
                        break
                    except Exception:
                        v = None

            if v is None:
                continue
            if v > best_map:
                best_map = v
                best_epoch = epoch

    if best_epoch < 0:
        return float("nan"), -1
    return best_map, best_epoch


def discover_anchor_weights() -> list[Path]:
    preferred = [
        ROOT / "runs/GAIIC2024_n_arch/n-arch-n-EFM-baseline/weights/best.pt",
        ROOT / "runs/GAIIC2024_archscan/archscan-RGBT-midfusion-EFM/weights/best.pt",
        ROOT / "runs/GAIIC2024_archscan/archscan-RGBT-midfusion/weights/best.pt",
        ROOT / "runs/GAIIC2024_archscan/archscan-RGBT-midfusion-P3/weights/best.pt",
    ]
    out = [p.resolve() for p in preferred if p.exists()]
    if out:
        return out

    fallback = sorted(
        (ROOT / "runs").glob("**/weights/best.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return [p.resolve() for p in fallback[:5]]


def parse_official_leaderboard(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec: dict[str, Any] = dict(row)
            try:
                rec["official_val_map50_95"] = float(row.get("official_val_map50_95", "nan"))
            except Exception:
                rec["official_val_map50_95"] = float("nan")
            rows.append(rec)
    return rows


def fallback_anchor_map_from_weights(weights: list[Path]) -> float:
    vals = []
    for w in weights:
        results = w.parent.parent / "results.csv"
        m, _ = parse_best_map(results)
        if is_finite(m):
            vals.append(m)
    return max(vals) if vals else float("nan")


def train_once(
    *,
    model_source: Path,
    load_weights: Path | None,
    data_yaml: Path,
    imgsz: int,
    epochs: int,
    project: Path,
    name: str,
    device: str,
    batch: int,
    workers: int,
    optimizer: str,
    close_mosaic: int,
    seed: int,
    mosaic: float | None = None,
    mixup: float | None = None,
    copy_paste: float | None = None,
    lr0: float | None = None,
    lrf: float | None = None,
    momentum: float | None = None,
    weight_decay: float | None = None,
    patience: int | None = None,
) -> dict[str, Any]:
    YOLO = get_yolo_cls()
    model = YOLO(str(model_source))
    if load_weights is not None:
        model.load(str(load_weights))

    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "cache": False,
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "close_mosaic": close_mosaic,
        "workers": workers,
        "device": device,
        "optimizer": optimizer,
        "use_simotm": "RGBT",
        "channels": 4,
        "pairs_rgb_ir": ["rgb", "tir"],
        "project": str(project),
        "name": name,
        "seed": seed,
        "exist_ok": False,
    }
    if mosaic is not None:
        kwargs["mosaic"] = mosaic
    if mixup is not None:
        kwargs["mixup"] = mixup
    if copy_paste is not None:
        kwargs["copy_paste"] = copy_paste
    if lr0 is not None:
        kwargs["lr0"] = lr0
    if lrf is not None:
        kwargs["lrf"] = lrf
    if momentum is not None:
        kwargs["momentum"] = momentum
    if weight_decay is not None:
        kwargs["weight_decay"] = weight_decay
    if patience is not None:
        kwargs["patience"] = patience

    model.train(**kwargs)

    run_dir = project / name
    ckpt = get_ckpt(run_dir)
    best_map, best_epoch = parse_best_map(run_dir / "results.csv")
    return {
        "run_dir": str(run_dir),
        "weights": str(ckpt.resolve()),
        "best_map50_95": best_map,
        "best_epoch": best_epoch,
        "imgsz": imgsz,
        "epochs": epochs,
    }


def run_branch(
    *,
    tag: str,
    model_cfg: Path,
    imgsz: int,
    seed_weights: Path,
    anchor_map: float,
    args: argparse.Namespace,
    run_root: Path,
) -> dict[str, Any]:
    branch_out: dict[str, Any] = {
        "tag": tag,
        "model": str(model_cfg),
        "imgsz": imgsz,
        "status": "started",
    }

    pilot_name = f"{tag.lower()}-pilot{args.pilot_epochs}-img{imgsz}"
    pilot = train_once(
        model_source=model_cfg,
        load_weights=seed_weights,
        data_yaml=resolve_path(args.data),
        imgsz=imgsz,
        epochs=args.pilot_epochs,
        project=run_root / "e2e4",
        name=pilot_name,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        optimizer=args.optimizer,
        close_mosaic=args.close_mosaic,
        seed=args.seed,
    )
    branch_out["pilot"] = pilot

    pilot_map = float(pilot["best_map50_95"])
    if (not args.disable_rule1) and is_finite(anchor_map) and is_finite(pilot_map) and pilot_map < anchor_map - args.early_drop_gap:
        branch_out["status"] = "stopped_by_rule1"
        branch_out["stop_reason"] = (
            f"pilot_map({pilot_map:.6f}) < anchor_map({anchor_map:.6f}) - early_drop_gap({args.early_drop_gap:.6f})"
        )
        branch_out["best_map50_95"] = pilot_map
        branch_out["best_weights"] = pilot["weights"]
        branch_out["qualified_for_e5"] = False
        return branch_out

    remain = max(0, args.full_epochs - args.pilot_epochs)
    if remain > 0:
        full_name = f"{tag.lower()}-cont{remain}-img{imgsz}"
        cont = train_once(
            model_source=Path(pilot["weights"]),
            load_weights=None,
            data_yaml=resolve_path(args.data),
            imgsz=imgsz,
            epochs=remain,
            project=run_root / "e2e4",
            name=full_name,
            device=args.device,
            batch=args.batch,
            workers=args.workers,
            optimizer=args.optimizer,
            close_mosaic=args.close_mosaic,
            seed=args.seed,
        )
        branch_out["continue"] = cont
        cont_map = float(cont["best_map50_95"])
        if is_finite(cont_map) and (not is_finite(pilot_map) or cont_map >= pilot_map):
            best_map = cont_map
            best_weights = cont["weights"]
        else:
            best_map = pilot_map
            best_weights = pilot["weights"]
    else:
        best_map = pilot_map
        best_weights = pilot["weights"]

    gain = best_map - anchor_map if is_finite(best_map) and is_finite(anchor_map) else float("nan")
    if args.disable_rule2:
        qualified = True
    else:
        qualified = is_finite(gain) and gain >= args.min_gain

    branch_out["status"] = "completed"
    branch_out["best_map50_95"] = best_map
    branch_out["best_weights"] = best_weights
    branch_out["gain_vs_anchor"] = gain
    branch_out["qualified_for_e5"] = qualified
    if (not qualified) and (not args.disable_rule2):
        branch_out["rule2_block_reason"] = (
            f"gain({gain:.6f}) < min_gain({args.min_gain:.6f})" if is_finite(gain) else "gain is NaN"
        )
    return branch_out


def select_best_branch(branches: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [b for b in branches if is_finite(float(b.get("best_map50_95", float("nan"))))]
    if not valid:
        return None
    return max(valid, key=lambda x: float(x["best_map50_95"]))


def run_e5(
    *,
    initial_weights: Path,
    imgsz: int,
    args: argparse.Namespace,
    run_root: Path,
) -> dict[str, Any]:
    e5_project = run_root / "e5"
    e5_name = "balanced-hardneg-kfold"
    cmd = [
        sys.executable,
        str((ROOT / "tools/run_balanced_hardneg_kfold_pipeline.py").resolve()),
        "--dataset-root",
        str(resolve_path(args.dataset_root)),
        "--initial-weights",
        str(initial_weights.resolve()),
        "--folds",
        str(args.e5_folds),
        "--rounds",
        str(args.e5_rounds),
        "--base-epochs",
        str(args.e5_base_epochs),
        "--online-epochs",
        str(args.e5_online_epochs),
        "--imgsz",
        str(imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
        "--project",
        str(e5_project),
        "--name",
        e5_name,
        "--seed",
        str(args.seed),
    ]
    run_cmd(cmd, cwd=ROOT)

    e5_run_dir = e5_project / e5_name
    best_fold_json = e5_run_dir / "best_fold.json"
    out: dict[str, Any] = {
        "run_dir": str(e5_run_dir),
        "best_map50_95": float("nan"),
        "best_weights": "",
    }

    if best_fold_json.exists():
        obj = json.loads(best_fold_json.read_text(encoding="utf-8"))
        out["best_map50_95"] = float(obj.get("best_map50_95", float("nan")))
        out["best_weights"] = str(obj.get("best_weights", ""))
        out["best_fold"] = obj.get("fold")
        out["best_round"] = obj.get("best_round")

    if out["best_weights"]:
        w = Path(str(out["best_weights"]))
        if not w.is_absolute():
            w = (ROOT / w).resolve()
        out["best_weights"] = str(w)

    return out


def run_e6(
    *,
    start_weights: Path,
    args: argparse.Namespace,
    run_root: Path,
) -> dict[str, Any]:
    rec = train_once(
        model_source=start_weights,
        load_weights=None,
        data_yaml=resolve_path(args.stage2_data),
        imgsz=args.e6_imgsz,
        epochs=args.e6_epochs,
        project=run_root / "e6",
        name="stage2-hardneg",
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        optimizer="SGD",
        close_mosaic=0,
        seed=args.seed,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        lr0=0.002,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        patience=20,
    )
    return rec


def save_summary(run_root: Path, summary: dict[str, Any]) -> Path:
    summary_path = run_root / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def prepare_run_root(project: Path, run_name: str, strict: bool) -> tuple[Path, str, bool]:
    run_root = project / run_name
    if not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=False)
        return run_root, run_name, False

    if strict:
        raise FileExistsError(f"Run directory already exists: {run_root}")

    for i in range(1, 1000):
        alt_name = f"{run_name}-r{i:02d}"
        alt_root = project / alt_name
        if not alt_root.exists():
            alt_root.mkdir(parents=True, exist_ok=False)
            return alt_root, alt_name, True

    raise FileExistsError(
        f"Unable to allocate unique run directory under {project} for base name '{run_name}'"
    )


def main() -> None:
    args = parse_args()

    args.dataset_root = resolve_path(args.dataset_root)
    args.data = resolve_path(args.data)
    args.stage2_data = resolve_path(args.stage2_data)
    args.e2_model = resolve_path(args.e2_model)
    args.e3_model = resolve_path(args.e3_model)
    args.e4_model = resolve_path(args.e4_model)
    args.project = resolve_path(args.project)
    selected_tags = parse_run_tags(args.run_tags)
    if args.anchor_source_summary is not None:
        args.anchor_source_summary = resolve_path(args.anchor_source_summary)

    if args.timestamp:
        run_name = f"{args.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    else:
        run_name = args.name
    run_root, resolved_run_name, renamed = prepare_run_root(args.project, run_name, args.strict_run_root)

    summary: dict[str, Any] = {
        "run_root": str(run_root),
        "run_name": resolved_run_name,
        "run_name_requested": run_name,
        "run_name_auto_renamed": renamed,
        "argv": sys.argv,
        "selected_tags": selected_tags,
        "branch_only": bool(args.branch_only),
        "stop_rules": {
            "rule1": f"pilot_map < anchor_map - {args.early_drop_gap}",
            "rule1_enabled": not args.disable_rule1,
            "rule2": f"gain_to_next_stage >= {args.min_gain}",
            "rule2_enabled": not args.disable_rule2,
        },
    }

    log(
        "[CONFIG] "
        f"run_tags={selected_tags} "
        f"branch_only={args.branch_only} "
        f"skip_e1={args.skip_e1} "
        f"disable_rule1={args.disable_rule1} "
        f"disable_rule2={args.disable_rule2}"
    )
    if selected_tags == list(VALID_BRANCH_TAGS) and (not args.branch_only):
        log("[WARN] Using default run-tags E2,E3,E4 with full pipeline. For single-branch runs, set --run-tags and --branch-only.")
    if renamed:
        log(f"[WARN] run directory exists, auto-renamed to: {run_root}")

    if args.anchor_weights:
        anchors = [resolve_path(Path(x)) for x in args.anchor_weights]
    else:
        anchors = discover_anchor_weights()

    anchors = [x for x in anchors if x.exists()]
    anchor_map = float("nan")
    e1_best_weight: Path | None = None

    if args.skip_e1:
        if args.anchor_source_summary is not None:
            if not args.anchor_source_summary.exists():
                raise FileNotFoundError(f"anchor-source-summary not found: {args.anchor_source_summary}")
            loaded_map, loaded_seed = load_anchor_from_summary(args.anchor_source_summary)
            if is_finite(loaded_map):
                anchor_map = loaded_map
            if loaded_seed is not None:
                e1_best_weight = loaded_seed

        if args.anchor_map is not None:
            anchor_map = float(args.anchor_map)

        if (not is_finite(anchor_map)) and anchors:
            anchor_map = fallback_anchor_map_from_weights(anchors)

        summary["e1"] = {
            "status": "skipped",
            "reason": "--skip-e1",
            "anchor_source_summary": str(args.anchor_source_summary) if args.anchor_source_summary is not None else "",
            "anchor_weights": [str(x) for x in anchors],
            "anchor_map50_95": anchor_map,
        }
    else:
        if not anchors:
            raise FileNotFoundError("No anchor weights found. Please provide --anchor-weights")

        log("[E1] official recheck starting")
        e1_out = run_root / "e1_official"
        e1_cmd = [
            sys.executable,
            str((ROOT / "tools/recheck_official_val.py").resolve()),
            "--weights",
            *[str(x) for x in anchors],
            "--dataset-root",
            str(args.dataset_root),
            "--imgsz",
            str(args.e1_imgsz),
            "--batch",
            str(args.batch),
            "--workers",
            str(args.workers),
            "--device",
            str(args.device),
            "--out-dir",
            str(e1_out),
        ]
        run_cmd(e1_cmd, cwd=ROOT)

        e1_rows = parse_official_leaderboard(e1_out / "official_val_leaderboard.csv")
        if e1_rows:
            top = e1_rows[0]
            anchor_map = float(top.get("official_val_map50_95", float("nan")))
            w = Path(str(top.get("weights", "")))
            if w.exists():
                e1_best_weight = w.resolve()

        if not is_finite(anchor_map):
            anchor_map = fallback_anchor_map_from_weights(anchors)

        summary["e1"] = {
            "status": "completed",
            "anchor_weights": [str(x) for x in anchors],
            "anchor_map50_95": anchor_map,
            "leaderboard_csv": str(e1_out / "official_val_leaderboard.csv"),
        }

    if args.seed_weights is not None:
        seed_weights = resolve_path(args.seed_weights)
    elif e1_best_weight is not None:
        seed_weights = e1_best_weight
    elif args.anchor_source_summary is not None and args.anchor_source_summary.exists():
        _, loaded_seed = load_anchor_from_summary(args.anchor_source_summary)
        if loaded_seed is not None:
            seed_weights = loaded_seed
        elif anchors:
            seed_weights = anchors[0]
        else:
            raise FileNotFoundError("seed weights not found and no usable anchor weights")
    else:
        if not anchors:
            raise FileNotFoundError("seed weights not found. Please provide --seed-weights")
        seed_weights = anchors[0]
    if not seed_weights.exists():
        raise FileNotFoundError(f"seed weights not found: {seed_weights}")

    summary["e1"]["seed_weights"] = str(seed_weights)

    if is_finite(anchor_map):
        log(f"[E1] anchor_map50_95={anchor_map:.6f} seed={seed_weights}")
    else:
        log(f"[E1] anchor_map50_95=nan seed={seed_weights}")

    log("[E2-E4] branch training starting")
    branches: list[dict[str, Any]] = []
    branch_plan = {
        "E2": (args.e2_model, args.e2_imgsz),
        "E3": (args.e3_model, args.e3_imgsz),
        "E4": (args.e4_model, args.e4_imgsz),
    }
    for tag in VALID_BRANCH_TAGS:
        if tag not in selected_tags:
            branches.append(
                {
                    "tag": tag,
                    "status": "skipped_by_selection",
                    "reason": "not in --run-tags",
                    "qualified_for_e5": False,
                }
            )
            continue

        model_cfg, imgsz = branch_plan[tag]
        if not model_cfg.exists():
            branches.append(
                {
                    "tag": tag,
                    "status": "skipped",
                    "reason": f"model cfg missing: {model_cfg}",
                    "qualified_for_e5": False,
                }
            )
            continue
        log(f"[{tag}] start model={model_cfg.name} imgsz={imgsz}")
        rec = run_branch(
            tag=tag,
            model_cfg=model_cfg,
            imgsz=imgsz,
            seed_weights=seed_weights,
            anchor_map=anchor_map,
            args=args,
            run_root=run_root,
        )
        log(f"[{tag}] status={rec['status']} best_map50_95={rec.get('best_map50_95', float('nan'))}")
        branches.append(rec)

    summary["e2_e4"] = branches

    best_branch = select_best_branch(branches)
    summary["best_branch"] = best_branch

    if best_branch is None:
        summary["e5"] = {"status": "skipped", "reason": "no valid E2/E3/E4 branch"}
        summary["e6"] = {"status": "skipped", "reason": "E5 skipped"}
        save_summary(run_root, summary)
        log("[DONE] no valid branch, pipeline ended")
        return

    if args.branch_only:
        summary["e5"] = {"status": "skipped", "reason": "--branch-only"}
        summary["e6"] = {"status": "skipped", "reason": "--branch-only"}
        save_summary(run_root, summary)
        log("[DONE] --branch-only active, stopped after E2/E3/E4")
        return

    if (not args.disable_rule2) and (not bool(best_branch.get("qualified_for_e5", False))):
        summary["e5"] = {
            "status": "skipped",
            "reason": "blocked by rule2: best branch gain below min-gain",
        }
        summary["e6"] = {"status": "skipped", "reason": "E5 skipped"}
        save_summary(run_root, summary)
        log("[DONE] blocked by rule2 before E5")
        return

    log("[E5] balanced hardneg kfold starting")
    e5 = run_e5(
        initial_weights=Path(str(best_branch["best_weights"])).resolve(),
        imgsz=int(best_branch.get("imgsz", args.e2_imgsz)),
        args=args,
        run_root=run_root,
    )
    summary["e5"] = e5

    best_branch_map = float(best_branch.get("best_map50_95", float("nan")))
    e5_map = float(e5.get("best_map50_95", float("nan")))

    if (not args.disable_rule2) and (
        (not is_finite(e5_map)) or (not is_finite(best_branch_map)) or e5_map < best_branch_map + args.min_gain
    ):
        summary["e6"] = {
            "status": "skipped",
            "reason": (
                f"blocked by rule2: e5_map({e5_map:.6f}) < best_branch_map({best_branch_map:.6f}) + min_gain({args.min_gain:.6f})"
                if is_finite(e5_map) and is_finite(best_branch_map)
                else "blocked by rule2: invalid e5/best-branch map"
            ),
        }
        save_summary(run_root, summary)
        log("[DONE] blocked by rule2 before E6")
        return

    e5_weights = Path(str(e5.get("best_weights", "")))
    if not e5_weights.exists():
        summary["e6"] = {
            "status": "skipped",
            "reason": f"E5 best weights not found: {e5_weights}",
        }
        save_summary(run_root, summary)
        log("[DONE] E5 best weights missing, E6 skipped")
        return

    if not args.stage2_data.exists():
        summary["e6"] = {
            "status": "skipped",
            "reason": f"stage2 data yaml not found: {args.stage2_data}",
        }
        save_summary(run_root, summary)
        log("[DONE] stage2 yaml missing, E6 skipped")
        return

    log("[E6] stage2 hardneg fine-tune starting")
    e6 = run_e6(start_weights=e5_weights.resolve(), args=args, run_root=run_root)
    summary["e6"] = {
        "status": "completed",
        **e6,
    }

    summary_path = save_summary(run_root, summary)
    log(f"[DONE] summary saved: {summary_path}")


if __name__ == "__main__":
    main()
