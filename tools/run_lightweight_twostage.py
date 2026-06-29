#!/usr/bin/env python3
"""Run lightweight two-stage RGBT training with optional modal-shift augmentation.

Stage1 uses stronger augmentation to improve robustness.
Stage2 reloads stage1 best weight and trains with milder augmentation for final convergence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage lightweight YOLO11-RGBT training runner")
    parser.add_argument("--model", type=str, required=True, help="Model yaml path")
    parser.add_argument("--data", type=str, default="ultralytics/cfg/datasets/GAIIC2024-rgbt.yaml")
    parser.add_argument("--initial-weights", type=str, default="", help="Optional initial .pt weights")

    parser.add_argument("--project", type=str, default="runs/GAIIC2024_lightweight")
    parser.add_argument("--name", type=str, required=True)

    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", type=str, default="SGD")

    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage2-epochs", type=int, default=70)
    parser.add_argument("--stage2-lr-scale", type=float, default=0.5)

    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)

    parser.add_argument("--stage1-mosaic", type=float, default=1.0)
    parser.add_argument("--stage1-close-mosaic", type=int, default=0)
    parser.add_argument("--stage1-degrees", type=float, default=3.0)
    parser.add_argument("--stage1-translate", type=float, default=0.08)
    parser.add_argument("--stage1-scale", type=float, default=0.5)
    parser.add_argument("--stage1-flipud", type=float, default=0.15)
    parser.add_argument("--stage1-brightness", type=float, default=0.25)
    parser.add_argument("--stage1-modal-shift", type=float, default=0.25)
    parser.add_argument("--stage1-modal-shift-px", type=int, default=8)

    parser.add_argument("--stage2-mosaic", type=float, default=0.15)
    parser.add_argument("--stage2-close-mosaic", type=int, default=5)
    parser.add_argument("--stage2-degrees", type=float, default=1.0)
    parser.add_argument("--stage2-translate", type=float, default=0.04)
    parser.add_argument("--stage2-scale", type=float, default=0.3)
    parser.add_argument("--stage2-flipud", type=float, default=0.05)
    parser.add_argument("--stage2-brightness", type=float, default=0.15)
    parser.add_argument("--stage2-modal-shift", type=float, default=0.10)
    parser.add_argument("--stage2-modal-shift-px", type=int, default=4)

    parser.add_argument("--modal-shift-mode", type=str, default="rgb", choices=["rgb", "tir", "independent", "both"])

    parser.add_argument("--use-simotm", type=str, default="RGBT")
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--pairs-rgb", type=str, default="rgb")
    parser.add_argument("--pairs-ir", type=str, default="tir")

    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing run names")
    return parser.parse_args()


def pick_weight(run_dir: Path) -> Path:
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(f"No best.pt or last.pt found under {run_dir}")


def resolve_device(requested: str) -> str:
    import torch

    req = str(requested).strip()
    req_l = req.lower()

    if req_l in {"", "auto"}:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "0"
        return "cpu"

    if req_l == "cpu":
        return "cpu"

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError(
            "CUDA is unavailable in current environment while requesting GPU training. "
            f"torch.__version__={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
            f"cuda_available={torch.cuda.is_available()}, cuda_count={torch.cuda.device_count()}. "
            "Please check nvidia-smi and install CUDA-enabled torch in this conda env."
        )

    count = torch.cuda.device_count()
    parts: List[str] = [p.strip() for p in req.split(",") if p.strip()]
    if all(p.isdigit() for p in parts):
        ids = [int(p) for p in parts]
        bad = [i for i in ids if i < 0 or i >= count]
        if bad:
            if count == 1:
                return "0"
            raise ValueError(f"Invalid device '{req}'. Visible CUDA ids: 0..{count - 1}")

    return req


def train_one_stage(
    source: str,
    stage_name: str,
    common: dict,
    stage_overrides: dict,
    bootstrap_weights: str = "",
) -> tuple[Path, Path]:
    model = YOLO(source)
    if bootstrap_weights:
        model.load(bootstrap_weights)
    model.train(**common, **stage_overrides, name=stage_name)
    save_dir = Path(model.trainer.save_dir)
    ckpt = pick_weight(save_dir)
    return save_dir, ckpt


def main() -> None:
    args = parse_args()
    resolved_device = resolve_device(args.device)
    if resolved_device != args.device:
        print(f"[WARN] device auto-corrected from '{args.device}' to '{resolved_device}'")

    initial_weights = args.initial_weights.strip()
    if initial_weights and not Path(initial_weights).exists():
        raise FileNotFoundError(f"Initial weights not found: {initial_weights}")

    common = dict(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=resolved_device,
        seed=args.seed,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cache=False,
        use_simotm=args.use_simotm,
        channels=args.channels,
        pairs_rgb_ir=[args.pairs_rgb, args.pairs_ir],
        project=args.project,
        save_period=args.save_period,
        exist_ok=args.exist_ok,
    )

    stage1_name = f"{args.name}-s1"
    stage2_name = f"{args.name}-s2"

    stage1_overrides = dict(
        epochs=args.stage1_epochs,
        mosaic=args.stage1_mosaic,
        close_mosaic=max(0, args.stage1_close_mosaic),
        degrees=args.stage1_degrees,
        translate=args.stage1_translate,
        scale=args.stage1_scale,
        flipud=args.stage1_flipud,
        brightness=args.stage1_brightness,
        modal_shift=args.stage1_modal_shift,
        modal_shift_px=max(0, args.stage1_modal_shift_px),
        modal_shift_mode=args.modal_shift_mode,
    )

    stage2_close_mosaic = max(0, args.stage2_close_mosaic)
    if args.stage2_epochs > 0 and stage2_close_mosaic > args.stage2_epochs:
        stage2_close_mosaic = args.stage2_epochs

    stage2_overrides = dict(
        epochs=args.stage2_epochs,
        lr0=args.lr0 * args.stage2_lr_scale,
        mosaic=args.stage2_mosaic,
        close_mosaic=stage2_close_mosaic,
        degrees=args.stage2_degrees,
        translate=args.stage2_translate,
        scale=args.stage2_scale,
        flipud=args.stage2_flipud,
        brightness=args.stage2_brightness,
        modal_shift=args.stage2_modal_shift,
        modal_shift_px=max(0, args.stage2_modal_shift_px),
        modal_shift_mode=args.modal_shift_mode,
    )

    stage1_dir, stage1_ckpt = train_one_stage(
        args.model,
        stage1_name,
        common,
        stage1_overrides,
        bootstrap_weights=initial_weights,
    )

    if args.stage2_epochs > 0:
        stage2_dir, stage2_ckpt = train_one_stage(str(stage1_ckpt), stage2_name, common, stage2_overrides)
    else:
        stage2_dir, stage2_ckpt = stage1_dir, stage1_ckpt

    summary_root = Path(args.project) / args.name
    summary_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "name": args.name,
        "model": args.model,
        "data": args.data,
        "initial_weights": initial_weights,
        "stage1": {
            "run_dir": str(stage1_dir),
            "best_or_last": str(stage1_ckpt),
            "epochs": args.stage1_epochs,
            "settings": stage1_overrides,
        },
        "stage2": {
            "run_dir": str(stage2_dir),
            "best_or_last": str(stage2_ckpt),
            "epochs": args.stage2_epochs,
            "settings": stage2_overrides,
        },
        "final_weights": str(stage2_ckpt),
    }
    summary_path = summary_root / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[DONE] stage1_dir={stage1_dir}")
    print(f"[DONE] stage2_dir={stage2_dir}")
    print(f"[DONE] final_weights={stage2_ckpt}")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
