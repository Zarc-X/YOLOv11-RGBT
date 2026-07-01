#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path("/root/autodl-tmp/GAIIC2024-赛道1-目标检测任务")


def run_cmd(cmd: list[str]) -> None:
    print("[CMD] " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    root_path = str(ROOT)
    env["PYTHONPATH"] = root_path if not py_path else f"{root_path}:{py_path}"
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (ROOT / p).resolve()


def to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def load_run_args_from_weights(weights: Path) -> tuple[dict[str, Any], Path | None]:
    args_yaml = weights.resolve().parent.parent / "args.yaml"
    if not args_yaml.exists():
        return {}, None

    try:
        obj = yaml.safe_load(args_yaml.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] failed to parse run args: {args_yaml} ({e})", flush=True)
        return {}, args_yaml

    if isinstance(obj, dict):
        return obj, args_yaml
    return {}, args_yaml


def resolve_modal_config(
    args: argparse.Namespace,
    run_args: dict[str, Any],
) -> dict[str, Any]:
    pairs = run_args.get("pairs_rgb_ir")
    default_rgb, default_ir = "rgb", "tir"
    if isinstance(pairs, list) and len(pairs) == 2 and all(isinstance(x, str) and x for x in pairs):
        default_rgb, default_ir = pairs[0], pairs[1]
    else:
        r_rgb = run_args.get("pairs_rgb")
        r_ir = run_args.get("pairs_ir")
        if isinstance(r_rgb, str) and r_rgb:
            default_rgb = r_rgb
        if isinstance(r_ir, str) and r_ir:
            default_ir = r_ir

    auto_use = str(run_args.get("use_simotm") or "RGBT")
    auto_channels = to_int(run_args.get("channels"), 4)

    use_simotm = args.use_simotm or auto_use
    channels = args.channels if args.channels > 0 else auto_channels
    pairs_rgb = args.pairs_rgb or default_rgb
    pairs_ir = args.pairs_ir or default_ir
    return {
        "use_simotm": use_simotm,
        "channels": channels,
        "pairs_rgb": pairs_rgb,
        "pairs_ir": pairs_ir,
    }


def train_stage2(
    *,
    weights: Path,
    data_yaml: Path,
    project: Path,
    name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    seed: int,
    modal_cfg: dict[str, Any],
    optimizer: str,
    lr0: float,
    lrf: float,
    momentum: float,
    weight_decay: float,
    close_mosaic: int,
    mosaic: float,
    mixup: float,
    copy_paste: float,
    patience: int,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "ultralytics is not installed in current python env. Activate training env first."
        ) from e

    model = YOLO(str(weights))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        workers=workers,
        device=device,
        project=str(project),
        name=name,
        exist_ok=False,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        close_mosaic=close_mosaic,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        patience=patience,
        seed=seed,
        use_simotm=str(modal_cfg["use_simotm"]),
        channels=int(modal_cfg["channels"]),
        pairs_rgb_ir=[str(modal_cfg["pairs_rgb"]), str(modal_cfg["pairs_ir"])],
    )

    run_dir = project / name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    ckpt = best if best.exists() else last
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")

    return {
        "run_dir": str(run_dir),
        "weights": str(ckpt.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E5 then E6 directly from an existing branch weight.")
    parser.add_argument("--initial-weights", type=Path, required=True, help="Existing best branch weights (e.g. E3 best.pt)")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--imgsz", type=int, default=768, help="Image size for E5")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--no-auto-modal",
        action="store_true",
        help="Disable loading use_simotm/channels/pairs from initial weights args.yaml",
    )
    parser.add_argument("--use-simotm", type=str, default="", help="Override modal mode, e.g. RGBT or RGBRGB6C")
    parser.add_argument("--channels", type=int, default=0, help="Override channel count. 0 means auto/default")
    parser.add_argument("--pairs-rgb", type=str, default="", help="Override RGB pair folder token")
    parser.add_argument("--pairs-ir", type=str, default="", help="Override IR pair folder token")

    parser.add_argument("--e5-folds", type=int, default=3)
    parser.add_argument("--e5-rounds", type=int, default=1)
    parser.add_argument("--e5-base-epochs", type=int, default=20)
    parser.add_argument("--e5-online-epochs", type=int, default=10)
    parser.add_argument("--e5-optimizer", type=str, default="SGD")
    parser.add_argument("--e5-lr0", type=float, default=0.002)
    parser.add_argument("--e5-lrf", type=float, default=0.01)
    parser.add_argument("--e5-momentum", type=float, default=0.937)
    parser.add_argument("--e5-weight-decay", type=float, default=0.0005)
    parser.add_argument("--e5-patience", type=int, default=20)
    parser.add_argument("--e5-close-mosaic", type=int, default=0)
    parser.add_argument("--e5-mosaic", type=float, default=0.0)
    parser.add_argument("--e5-mixup", type=float, default=0.0)
    parser.add_argument("--e5-copy-paste", type=float, default=0.0)

    parser.add_argument("--run-e6", action="store_true", help="Run E6 stage2 after E5")
    parser.add_argument(
        "--stage2-data",
        type=Path,
        default=Path("runs/analysis/stage2_hardneg_midfusion/GAIIC2024-rgbt-stage2-hardneg.yaml"),
    )
    parser.add_argument("--e6-epochs", type=int, default=30)
    parser.add_argument("--e6-imgsz", type=int, default=640)
    parser.add_argument("--e6-optimizer", type=str, default="SGD")
    parser.add_argument("--e6-lr0", type=float, default=0.002)
    parser.add_argument("--e6-lrf", type=float, default=0.01)
    parser.add_argument("--e6-momentum", type=float, default=0.937)
    parser.add_argument("--e6-weight-decay", type=float, default=0.0005)
    parser.add_argument("--e6-patience", type=int, default=20)
    parser.add_argument("--e6-close-mosaic", type=int, default=0)
    parser.add_argument("--e6-mosaic", type=float, default=0.0)
    parser.add_argument("--e6-mixup", type=float, default=0.0)
    parser.add_argument("--e6-copy-paste", type=float, default=0.0)

    parser.add_argument("--project", type=Path, default=Path("runs/GAIIC2024_e5e6"))
    parser.add_argument("--name", type=str, default="e5e6-from-branch")
    parser.add_argument("--strict-run-root", action="store_true", help="Fail if run root already exists")
    return parser.parse_args()


def prepare_run_root(project: Path, name: str, strict: bool) -> tuple[Path, str, bool]:
    run_root = project / name
    if not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=False)
        return run_root, name, False

    if strict:
        raise FileExistsError(f"Run directory already exists: {run_root}")

    for i in range(1, 1000):
        alt = project / f"{name}-r{i:02d}"
        if not alt.exists():
            alt.mkdir(parents=True, exist_ok=False)
            return alt, alt.name, True
    raise FileExistsError(f"Unable to allocate unique run directory under {project}")


def main() -> None:
    args = parse_args()

    init_weights = resolve_path(args.initial_weights)
    if not init_weights.exists():
        raise FileNotFoundError(f"initial-weights not found: {init_weights}")

    args.dataset_root = resolve_path(args.dataset_root)
    args.project = resolve_path(args.project)
    args.stage2_data = resolve_path(args.stage2_data)
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset-root not found: {args.dataset_root}")
    run_root, resolved_name, renamed = prepare_run_root(args.project, args.name, args.strict_run_root)

    run_args: dict[str, Any] = {}
    run_args_path: Path | None = None
    if not args.no_auto_modal:
        run_args, run_args_path = load_run_args_from_weights(init_weights)
    modal_cfg = resolve_modal_config(args, run_args)
    print(
        "[INFO] modal "
        f"use_simotm={modal_cfg['use_simotm']} channels={modal_cfg['channels']} "
        f"pairs={modal_cfg['pairs_rgb']},{modal_cfg['pairs_ir']}",
        flush=True,
    )
    if run_args_path is not None:
        print(f"[INFO] modal source args.yaml: {run_args_path}", flush=True)

    summary: dict[str, Any] = {
        "run_root": str(run_root),
        "name": resolved_name,
        "auto_renamed": renamed,
        "initial_weights": str(init_weights),
        "modal_config": {
            **modal_cfg,
            "auto_from_weights": not args.no_auto_modal,
            "source_args_yaml": str(run_args_path) if run_args_path else "",
        },
        "tuning": {
            "e5": {
                "optimizer": args.e5_optimizer,
                "lr0": args.e5_lr0,
                "lrf": args.e5_lrf,
                "momentum": args.e5_momentum,
                "weight_decay": args.e5_weight_decay,
                "patience": args.e5_patience,
                "close_mosaic": args.e5_close_mosaic,
                "mosaic": args.e5_mosaic,
                "mixup": args.e5_mixup,
                "copy_paste": args.e5_copy_paste,
            },
            "e6": {
                "optimizer": args.e6_optimizer,
                "lr0": args.e6_lr0,
                "lrf": args.e6_lrf,
                "momentum": args.e6_momentum,
                "weight_decay": args.e6_weight_decay,
                "patience": args.e6_patience,
                "close_mosaic": args.e6_close_mosaic,
                "mosaic": args.e6_mosaic,
                "mixup": args.e6_mixup,
                "copy_paste": args.e6_copy_paste,
            },
        },
    }
    if renamed:
        print(f"[WARN] run root exists, auto-renamed to: {run_root}", flush=True)

    e5_project = run_root / "e5"
    e5_name = "balanced-hardneg-kfold"
    cmd = [
        sys.executable,
        str((ROOT / "tools/run_balanced_hardneg_kfold_pipeline.py").resolve()),
        "--dataset-root",
        str(args.dataset_root),
        "--initial-weights",
        str(init_weights),
        "--folds",
        str(args.e5_folds),
        "--rounds",
        str(args.e5_rounds),
        "--base-epochs",
        str(args.e5_base_epochs),
        "--online-epochs",
        str(args.e5_online_epochs),
        "--imgsz",
        str(args.imgsz),
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
        "--optimizer",
        str(args.e5_optimizer),
        "--lr0",
        str(args.e5_lr0),
        "--lrf",
        str(args.e5_lrf),
        "--momentum",
        str(args.e5_momentum),
        "--weight-decay",
        str(args.e5_weight_decay),
        "--patience",
        str(args.e5_patience),
        "--close-mosaic",
        str(args.e5_close_mosaic),
        "--mosaic",
        str(args.e5_mosaic),
        "--mixup",
        str(args.e5_mixup),
        "--copy-paste",
        str(args.e5_copy_paste),
        "--use-simotm",
        str(modal_cfg["use_simotm"]),
        "--channels",
        str(modal_cfg["channels"]),
        "--pairs-rgb",
        str(modal_cfg["pairs_rgb"]),
        "--pairs-ir",
        str(modal_cfg["pairs_ir"]),
    ]
    run_cmd(cmd)

    e5_run_dir = e5_project / e5_name
    best_fold = e5_run_dir / "best_fold.json"
    if not best_fold.exists():
        raise FileNotFoundError(f"best_fold.json missing: {best_fold}")
    obj = json.loads(best_fold.read_text(encoding="utf-8"))

    e5_best_w = Path(str(obj.get("best_weights", "")))
    if not e5_best_w.is_absolute():
        e5_best_w = (ROOT / e5_best_w).resolve()
    if not e5_best_w.exists():
        raise FileNotFoundError(f"E5 best weights not found: {e5_best_w}")

    summary["e5"] = {
        "run_dir": str(e5_run_dir),
        "best_fold": obj.get("fold"),
        "best_round": obj.get("best_round"),
        "best_map50_95": obj.get("best_map50_95"),
        "best_weights": str(e5_best_w),
    }

    if args.run_e6:
        if not args.stage2_data.exists():
            raise FileNotFoundError(f"stage2-data not found: {args.stage2_data}")
        e6 = train_stage2(
            weights=e5_best_w,
            data_yaml=args.stage2_data,
            project=run_root / "e6",
            name="stage2-hardneg",
            epochs=args.e6_epochs,
            imgsz=args.e6_imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            seed=args.seed,
              modal_cfg=modal_cfg,
              optimizer=args.e6_optimizer,
              lr0=args.e6_lr0,
              lrf=args.e6_lrf,
              momentum=args.e6_momentum,
              weight_decay=args.e6_weight_decay,
              close_mosaic=args.e6_close_mosaic,
              mosaic=args.e6_mosaic,
              mixup=args.e6_mixup,
              copy_paste=args.e6_copy_paste,
              patience=args.e6_patience,
        )
        summary["e6"] = {
            "status": "completed",
            **e6,
        }
    else:
        summary["e6"] = {
            "status": "skipped",
            "reason": "--run-e6 not set",
        }

    summary_path = run_root / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] summary: {summary_path}")


if __name__ == "__main__":
    main()
