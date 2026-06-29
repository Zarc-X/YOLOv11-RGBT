#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BASIC_RGBT_ARCHS = [
    "yolo11-RGBT-earlyfusion.yaml",
    "yolo11-RGBT-midfusion.yaml",
    "yolo11-RGBT-midfusion-P3.yaml",
    "yolo11-RGBT-mid-to-late-fusion.yaml",
    "yolo11-RGBT-latefusion.yaml",
    "yolo11-RGBT-scorefusion.yaml",
    "yolo11-RGBT-share.yaml",
]

RECOMMENDED_RGBT_ARCHS = [
    "yolo11-RGBT-midfusion-P3.yaml",
    "yolo11-RGBT-midfusion.yaml",
    "yolo11-RGBT-share.yaml",
    "yolo11-RGBT-earlyfusion.yaml",
    "yolo11-RGBT-mid-to-late-fusion.yaml",
    "yolo11-RGBT-latefusion.yaml",
]

SPEED_FIRST_RGBT_ARCHS = [
    "yolo11-RGBT-midfusion-P3.yaml",
    "yolo11-RGBT-midfusion.yaml",
    "yolo11-RGBT-share.yaml",
]

BEST_KNOWN_RGBT_ARCHS = [
    "yolo11-RGBT-midfusion-P3.yaml",
]

DEFAULT_RGBT_EXCLUDES = [
    "seg",
    "pose",
    "obb",
    "MCF",
    "RGBRGB6C",
    "-10c",
    "-7c",
    "-test",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch architecture search with YOLO11n alias + pseudo-private/public-val scoring"
    )
    parser.add_argument("--base-data", type=Path, default=Path("ultralytics/cfg/datasets/GAIIC2024-rgbt.yaml"))
    parser.add_argument(
        "--public-val-data",
        type=Path,
        default=None,
        help="Public validation dataset yaml. Empty means same as --base-data.",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("ultralytics/cfg/models/11-RGBT"))
    parser.add_argument(
        "--arch-set",
        choices=["best-known", "speed-first", "recommended", "basic", "all-detect-rgbt", "manual"],
        default="recommended",
        help="Architecture candidate set. recommended is the practical default.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Manual model yaml list. Used when --arch-set=manual.",
    )
    parser.add_argument(
        "--exclude-keywords",
        nargs="*",
        default=DEFAULT_RGBT_EXCLUDES,
        help="Keywords to exclude when --arch-set=all-detect-rgbt.",
    )
    parser.add_argument("--max-arches", type=int, default=0, help="0 means all discovered architectures.")
    parser.add_argument("--dry-run", action="store_true", help="Only build split and print planned runs.")

    parser.add_argument("--project", type=Path, default=Path("runs/GAIIC2024_archscan"))
    parser.add_argument("--name-prefix", type=str, default="archscan-")
    parser.add_argument("--exist-ok", dest="exist_ok", action="store_true", help="Reuse existing run names.")
    parser.add_argument("--no-exist-ok", dest="exist_ok", action="store_false", help="Allow auto-incremented run names.")
    parser.set_defaults(exist_ok=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--patience", type=int, default=50)

    parser.add_argument("--use-simotm", type=str, default="RGBT")
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--pairs-rgb", type=str, default="rgb")
    parser.add_argument("--pairs-ir", type=str, default="tir")

    parser.add_argument("--pseudo-val-ratio", type=float, default=0.15)
    parser.add_argument("--pseudo-seed", type=int, default=2026)
    parser.add_argument("--rebuild-pseudo-split", action="store_true")

    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--pretrained-weights", type=str, default="", help="Optional weights path to load before train.")

    parser.add_argument("--score-pseudo-weight", type=float, default=0.7)
    parser.add_argument("--score-public-weight", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=5)

    return parser.parse_args()


def _norm_float(x: Any) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {k.strip(): v for k, v in row.items() if k is not None}


def _first_present(row: dict[str, str], keys: list[str]) -> str:
    for k in keys:
        if k in row:
            return row[k]
    return ""


def parse_results_csv(results_csv: Path) -> tuple[float, float, int]:
    """Return (best_map50_95, last_map50_95, epoch_rows)."""
    if not results_csv.exists():
        return float("nan"), float("nan"), 0

    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = [_normalize_row(r) for r in csv.DictReader(f)]

    if not rows:
        return float("nan"), float("nan"), 0

    key_options = ["metrics/mAP50-95(B)", "metrics/mAP50-95"]

    best_idx = 0
    best_val = -1.0
    for i, r in enumerate(rows):
        v = _norm_float(_first_present(r, key_options))
        if not math.isnan(v) and v > best_val:
            best_val = v
            best_idx = i

    last_val = _norm_float(_first_present(rows[-1], key_options))
    return _norm_float(best_val), last_val, len(rows)


def _stable_pick(rel_path: str, ratio: float, seed: int) -> bool:
    key = f"{seed}:{rel_path}".encode("utf-8")
    h = int(hashlib.md5(key).hexdigest(), 16)
    threshold = int(max(0.0, min(1.0, ratio)) * 100000)
    return (h % 100000) < threshold


def resolve_entry_path(entry: str, data_root: Path | None, yaml_parent: Path) -> Path:
    p = Path(entry)
    if p.is_absolute():
        return p

    candidates = []
    if data_root is not None:
        candidates.append((data_root / p).resolve())
    candidates.append((yaml_parent / p).resolve())

    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _read_txt_image_list(txt_path: Path, data_root: Path | None) -> list[Path]:
    images: list[Path] = []
    for raw in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        if not p.is_absolute():
            cand1 = (txt_path.parent / p).resolve()
            cand2 = ((data_root / p).resolve() if data_root is not None else cand1)
            p = cand1 if cand1.exists() else cand2
        if p.suffix.lower() in IMG_EXTS:
            images.append(p.resolve())
    return images


def _collect_images_from_entry(entry: Any, data_root: Path | None, yaml_parent: Path) -> list[Path]:
    if isinstance(entry, list):
        out: list[Path] = []
        for sub in entry:
            out.extend(_collect_images_from_entry(sub, data_root, yaml_parent))
        return sorted(set(out))

    if not isinstance(entry, str):
        raise TypeError(f"Unsupported train entry type: {type(entry)}")

    p = resolve_entry_path(entry, data_root, yaml_parent)
    if p.is_file() and p.suffix.lower() == ".txt":
        return sorted(set(_read_txt_image_list(p, data_root)))

    if p.is_dir():
        files = [x.resolve() for x in p.rglob("*") if x.is_file() and x.suffix.lower() in IMG_EXTS]
        return sorted(files)

    raise FileNotFoundError(f"Unable to resolve dataset entry: {entry} -> {p}")


def _relative_or_name(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except Exception:
            pass
    return path.name


def build_pseudo_split(
    base_data_yaml: Path,
    project_dir: Path,
    ratio: float,
    seed: int,
    rebuild: bool,
) -> tuple[Path, Path, Path, int, int]:
    base_data_yaml = base_data_yaml.resolve()
    data = yaml.safe_load(base_data_yaml.read_text(encoding="utf-8"))
    yaml_parent = base_data_yaml.parent

    data_root_raw = data.get("path", "")
    data_root = None
    if isinstance(data_root_raw, str) and data_root_raw:
        data_root = resolve_entry_path(data_root_raw, None, yaml_parent)

    train_entry = data.get("train")
    if train_entry is None:
        raise KeyError(f"Missing 'train' field in {base_data_yaml}")

    train_images = _collect_images_from_entry(train_entry, data_root, yaml_parent)
    if not train_images:
        raise RuntimeError("No train images found for pseudo split construction.")

    meta_dir = (project_dir / "_archscan_meta").resolve()
    meta_dir.mkdir(parents=True, exist_ok=True)

    ratio_tag = str(int(max(0.0, min(1.0, ratio)) * 1000)).zfill(3)
    split_tag = f"seed{seed}_r{ratio_tag}"
    pseudo_train_txt = meta_dir / f"pseudo_train_{split_tag}.txt"
    pseudo_val_txt = meta_dir / f"pseudo_val_{split_tag}.txt"
    pseudo_data_yaml = meta_dir / f"pseudo_data_{split_tag}.yaml"

    if rebuild or not (pseudo_train_txt.exists() and pseudo_val_txt.exists() and pseudo_data_yaml.exists()):
        pseudo_train: list[Path] = []
        pseudo_val: list[Path] = []

        for img in train_images:
            rel_key = _relative_or_name(img, data_root)
            if _stable_pick(rel_key, ratio, seed):
                pseudo_val.append(img)
            else:
                pseudo_train.append(img)

        if not pseudo_val:
            pseudo_val.append(pseudo_train.pop())
        if not pseudo_train:
            pseudo_train.append(pseudo_val.pop())

        pseudo_train_txt.write_text("\n".join(str(x) for x in pseudo_train) + "\n", encoding="utf-8")
        pseudo_val_txt.write_text("\n".join(str(x) for x in pseudo_val) + "\n", encoding="utf-8")

        pseudo_yaml_obj = {
            "path": str(data_root) if data_root is not None else "",
            "train": str(pseudo_train_txt),
            "val": str(pseudo_val_txt),
            "nc": data.get("nc"),
            "names": data.get("names"),
        }
        pseudo_data_yaml.write_text(yaml.safe_dump(pseudo_yaml_obj, sort_keys=False, allow_unicode=False), encoding="utf-8")

    train_count = len([x for x in pseudo_train_txt.read_text(encoding="utf-8").splitlines() if x.strip()])
    val_count = len([x for x in pseudo_val_txt.read_text(encoding="utf-8").splitlines() if x.strip()])
    return pseudo_data_yaml, pseudo_train_txt, pseudo_val_txt, train_count, val_count


def to_n_alias_path(model_path: Path) -> str:
    stem = model_path.stem
    stem_n = re.sub(r"^yolo11(?:[nslmx])?-", "yolo11n-", stem)
    return str(model_path.with_name(stem_n + model_path.suffix))


def discover_architectures(args: argparse.Namespace) -> list[Path]:
    model_dir = args.model_dir.resolve()

    if args.arch_set == "manual":
        if not args.models:
            raise ValueError("--arch-set=manual requires --models")
        return [Path(x).resolve() for x in args.models]

    if args.arch_set == "best-known":
        out = [model_dir / name for name in BEST_KNOWN_RGBT_ARCHS]
        miss = [str(x) for x in out if not x.exists()]
        if miss:
            raise FileNotFoundError(f"Missing best-known architecture files: {miss}")
        return out

    if args.arch_set == "speed-first":
        out = [model_dir / name for name in SPEED_FIRST_RGBT_ARCHS]
        miss = [str(x) for x in out if not x.exists()]
        if miss:
            raise FileNotFoundError(f"Missing speed-first architecture files: {miss}")
        return out

    if args.arch_set == "recommended":
        out = [model_dir / name for name in RECOMMENDED_RGBT_ARCHS]
        miss = [str(x) for x in out if not x.exists()]
        if miss:
            raise FileNotFoundError(f"Missing recommended architecture files: {miss}")
        return out

    if args.arch_set == "basic":
        out = [model_dir / name for name in BASIC_RGBT_ARCHS]
        miss = [str(x) for x in out if not x.exists()]
        if miss:
            raise FileNotFoundError(f"Missing basic architecture files: {miss}")
        return out

    all_rgbt = sorted(model_dir.glob("yolo11-RGBT-*.yaml"))
    excludes = [k.lower() for k in args.exclude_keywords]
    out = []
    for p in all_rgbt:
        n = p.name.lower()
        if any(k in n for k in excludes):
            continue
        out.append(p)

    if not out:
        raise RuntimeError("No architectures discovered after filtering.")
    return out


def evaluate_public_val(
    weights_path: Path,
    public_data_yaml: Path,
    args: argparse.Namespace,
) -> tuple[float, float]:
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(public_data_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        use_simotm=args.use_simotm,
        channels=args.channels,
        pairs_rgb_ir=[args.pairs_rgb, args.pairs_ir],
        save_json=False,
        plots=False,
        verbose=False,
    )
    public_map = float(getattr(metrics.box, "map", float("nan"))) if hasattr(metrics, "box") else float("nan")
    public_map50 = float(getattr(metrics.box, "map50", float("nan"))) if hasattr(metrics, "box") else float("nan")
    return _norm_float(public_map), _norm_float(public_map50)


def score_record(rec: dict[str, Any], w_pseudo: float, w_public: float) -> float:
    pm = _norm_float(rec.get("pseudo_best_map50_95", float("nan")))
    vm = _norm_float(rec.get("public_val_map50_95", float("nan")))
    if math.isnan(pm) or math.isnan(vm):
        return float("nan")

    total = max(1e-12, w_pseudo + w_public)
    wp = w_pseudo / total
    wv = w_public / total
    return wp * pm + wv * vm


def write_leaderboard(project_dir: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    scored = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        score = score_record(rec, args.score_pseudo_weight, args.score_public_weight)
        if math.isnan(score):
            continue
        rec2 = dict(rec)
        rec2["score"] = score
        scored.append(rec2)

    scored.sort(key=lambda x: x["score"], reverse=True)

    out_csv = project_dir / "archscan_leaderboard.csv"
    out_md = project_dir / "archscan_recommendation.md"

    headers = [
        "rank",
        "run_name",
        "arch_source",
        "arch_n_alias",
        "pseudo_best_map50_95",
        "public_val_map50_95",
        "score",
        "run_dir",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for idx, r in enumerate(scored, start=1):
            writer.writerow(
                [
                    idx,
                    r.get("run_name", ""),
                    r.get("arch_source", ""),
                    r.get("arch_n_alias", ""),
                    f"{_norm_float(r.get('pseudo_best_map50_95')):.6f}",
                    f"{_norm_float(r.get('public_val_map50_95')):.6f}",
                    f"{_norm_float(r.get('score')):.6f}",
                    r.get("run_dir", ""),
                ]
            )

    md_lines = [
        "# Architecture Recommendation",
        "",
        f"Weights: pseudo_private={args.score_pseudo_weight}, public_val={args.score_public_weight}",
        "",
        "| Rank | Run | Source Arch | n Alias | Pseudo mAP50-95 | Public mAP50-95 | Score |",
        "|---:|---|---|---|---:|---:|---:|",
    ]

    for idx, r in enumerate(scored[: max(1, args.topk)], start=1):
        md_lines.append(
            "| {rank} | {run} | {src} | {alias} | {pm:.6f} | {vm:.6f} | {sc:.6f} |".format(
                rank=idx,
                run=r.get("run_name", ""),
                src=Path(str(r.get("arch_source", ""))).name,
                alias=Path(str(r.get("arch_n_alias", ""))).name,
                pm=_norm_float(r.get("pseudo_best_map50_95")),
                vm=_norm_float(r.get("public_val_map50_95")),
                sc=_norm_float(r.get("score")),
            )
        )

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return scored


def main() -> None:
    args = parse_args()

    project_dir = args.project.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    base_data_yaml = args.base_data.resolve()
    public_data_yaml = args.public_val_data.resolve() if args.public_val_data else base_data_yaml

    pseudo_yaml, pseudo_train_txt, pseudo_val_txt, n_train, n_val = build_pseudo_split(
        base_data_yaml=base_data_yaml,
        project_dir=project_dir,
        ratio=args.pseudo_val_ratio,
        seed=args.pseudo_seed,
        rebuild=args.rebuild_pseudo_split,
    )

    arch_files = discover_architectures(args)
    if args.max_arches and args.max_arches > 0:
        arch_files = arch_files[: args.max_arches]

    summary_jsonl = project_dir / "archscan_summary.jsonl"
    summary_records: list[dict[str, Any]] = []

    print("[INFO] pseudo_data_yaml:", pseudo_yaml)
    print("[INFO] pseudo_train_list:", pseudo_train_txt, "count=", n_train)
    print("[INFO] pseudo_val_list:", pseudo_val_txt, "count=", n_val)
    print("[INFO] candidate_arch_count:", len(arch_files))

    if args.dry_run:
        print("[INFO] dry-run enabled. Planned architecture runs:")
        for idx, arch_path in enumerate(arch_files, start=1):
            arch_alias = to_n_alias_path(arch_path.resolve())
            short = re.sub(r"^yolo11(?:[nslmx])?-", "", arch_path.stem)
            run_name = f"{args.name_prefix}{short}"
            print(f"  {idx}. {arch_path.name} -> {Path(arch_alias).name} | run_name={run_name}")
        return

    for idx, arch_path in enumerate(arch_files, start=1):
        arch_path = arch_path.resolve()
        arch_alias = to_n_alias_path(arch_path)

        short = re.sub(r"^yolo11(?:[nslmx])?-", "", arch_path.stem)
        run_name = f"{args.name_prefix}{short}"
        run_dir = project_dir / run_name

        rec: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "index": idx,
            "total": len(arch_files),
            "arch_source": str(arch_path),
            "arch_n_alias": str(arch_alias),
            "run_name": run_name,
            "run_dir": str(run_dir),
            "pseudo_data_yaml": str(pseudo_yaml),
            "public_val_data_yaml": str(public_data_yaml),
            "status": "unknown",
        }

        print(f"\n[RUN {idx}/{len(arch_files)}] {arch_path.name} -> alias {Path(arch_alias).name}")

        try:
            if args.resume_existing and (run_dir / "weights" / "best.pt").exists() and (run_dir / "results.csv").exists():
                print("[INFO] resume-existing enabled, skip train:", run_name)
            else:
                model = YOLO(arch_alias)
                if args.pretrained_weights:
                    weights_path = Path(args.pretrained_weights)
                    if weights_path.exists():
                        model.load(str(weights_path))
                    else:
                        print(f"[WARN] pretrained weights not found, skip load: {weights_path}")

                model.train(
                    data=str(pseudo_yaml),
                    cache=False,
                    imgsz=args.imgsz,
                    epochs=args.epochs,
                    batch=args.batch,
                    close_mosaic=args.close_mosaic,
                    workers=args.workers,
                    device=args.device,
                    optimizer=args.optimizer,
                    patience=args.patience,
                    use_simotm=args.use_simotm,
                    channels=args.channels,
                    pairs_rgb_ir=[args.pairs_rgb, args.pairs_ir],
                    project=str(project_dir),
                    name=run_name,
                    exist_ok=args.exist_ok,
                    resume=False,
                )

                # Ultralytics may auto-increment run directory names when exist_ok=False.
                trainer = getattr(model, "trainer", None)
                save_dir = getattr(trainer, "save_dir", None)
                if save_dir:
                    run_dir = Path(str(save_dir)).resolve()
                    rec["run_dir"] = str(run_dir)
                    rec["run_name"] = run_dir.name

                del model
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            best_pt = run_dir / "weights" / "best.pt"
            results_csv = run_dir / "results.csv"
            pseudo_best, pseudo_last, epochs_recorded = parse_results_csv(results_csv)

            rec["pseudo_best_map50_95"] = pseudo_best
            rec["pseudo_last_map50_95"] = pseudo_last
            rec["epochs_recorded"] = epochs_recorded
            rec["has_best"] = best_pt.exists()

            if not best_pt.exists():
                raise FileNotFoundError(f"best.pt not found for run: {run_name}")

            public_map, public_map50 = evaluate_public_val(best_pt, public_data_yaml, args)
            rec["public_val_map50_95"] = public_map
            rec["public_val_map50"] = public_map50

            public_json = {
                "public_val_map50_95": public_map,
                "public_val_map50": public_map50,
                "evaluated_at": datetime.now().isoformat(timespec="seconds"),
                "weights": str(best_pt),
                "public_val_data": str(public_data_yaml),
            }
            (run_dir / "archscan_public_val.json").write_text(
                json.dumps(public_json, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            rec["status"] = "ok"
            rec["score"] = score_record(rec, args.score_pseudo_weight, args.score_public_weight)

        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()
            print("[ERROR]", rec["error"])

        summary_records.append(rec)
        with summary_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    leaderboard = write_leaderboard(project_dir, summary_records, args)

    print("\n[SUMMARY] summary_jsonl:", summary_jsonl)
    print("[SUMMARY] leaderboard_csv:", project_dir / "archscan_leaderboard.csv")
    print("[SUMMARY] recommendation_md:", project_dir / "archscan_recommendation.md")

    if leaderboard:
        print(f"[SUMMARY] Top-{min(args.topk, len(leaderboard))} recommendations:")
        for i, r in enumerate(leaderboard[: args.topk], start=1):
            print(
                f"  {i}. {r['run_name']} | score={r['score']:.6f} | "
                f"pseudo={_norm_float(r.get('pseudo_best_map50_95')):.6f} | "
                f"public={_norm_float(r.get('public_val_map50_95')):.6f}"
            )
    else:
        print("[SUMMARY] No successful scored runs.")


if __name__ == "__main__":
    main()
