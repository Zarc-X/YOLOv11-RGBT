#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str]) -> None:
    print("[CMD] " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    root_path = str(ROOT)
    env["PYTHONPATH"] = root_path if not py_path else f"{root_path}:{py_path}"
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (ROOT / p).resolve()


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
        optimizer="SGD",
        lr0=0.002,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        close_mosaic=0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        patience=20,
        seed=seed,
        use_simotm="RGBT",
        channels=4,
        pairs_rgb_ir=["rgb", "tir"],
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
    parser.add_argument("--imgsz", type=int, default=768, help="Image size for E5")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--e5-folds", type=int, default=3)
    parser.add_argument("--e5-rounds", type=int, default=1)
    parser.add_argument("--e5-base-epochs", type=int, default=20)
    parser.add_argument("--e5-online-epochs", type=int, default=10)

    parser.add_argument("--run-e6", action="store_true", help="Run E6 stage2 after E5")
    parser.add_argument(
        "--stage2-data",
        type=Path,
        default=Path("runs/analysis/stage2_hardneg_midfusion/GAIIC2024-rgbt-stage2-hardneg.yaml"),
    )
    parser.add_argument("--e6-epochs", type=int, default=30)
    parser.add_argument("--e6-imgsz", type=int, default=640)

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

    args.project = resolve_path(args.project)
    args.stage2_data = resolve_path(args.stage2_data)
    run_root, resolved_name, renamed = prepare_run_root(args.project, args.name, args.strict_run_root)

    summary: dict[str, Any] = {
        "run_root": str(run_root),
        "name": resolved_name,
        "auto_renamed": renamed,
        "initial_weights": str(init_weights),
    }
    if renamed:
        print(f"[WARN] run root exists, auto-renamed to: {run_root}", flush=True)

    e5_project = run_root / "e5"
    e5_name = "balanced-hardneg-kfold"
    cmd = [
        sys.executable,
        str((ROOT / "tools/run_balanced_hardneg_kfold_pipeline.py").resolve()),
        "--dataset-root",
        "/root/autodl-tmp/GAIIC2024-赛道1-目标检测任务",
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
