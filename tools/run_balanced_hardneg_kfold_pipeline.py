#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run category-balanced training + online hard-negative loops + k-fold model selection "
            "for GAIIC RGBT."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/root/autodl-tmp/GAIIC2024-赛道1-目标检测任务"),
        help="Dataset root directory",
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=None,
        help="COCO train annotation json. Default: <dataset-root>/train/train.json",
    )
    parser.add_argument(
        "--initial-weights",
        type=Path,
        default=Path("runs/GAIIC2024_stage2/stage2-midfusion-hardneg3/weights/best.pt"),
        help="Initial checkpoint used for fold round-0 training",
    )
    parser.add_argument("--folds", type=int, default=3, help="Number of folds")
    parser.add_argument("--rounds", type=int, default=2, help="Hard-negative online rounds per fold")
    parser.add_argument("--base-epochs", type=int, default=20, help="Epochs for round-0 on each fold")
    parser.add_argument("--online-epochs", type=int, default=10, help="Epochs for each online round (>0)")

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr0", type=float, default=0.002)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--close-mosaic", type=int, default=0)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)

    parser.add_argument("--use-simotm", type=str, default="RGBT")
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--pairs-rgb", type=str, default="rgb")
    parser.add_argument("--pairs-ir", type=str, default="tir")

    parser.add_argument("--balance-max-repeat", type=int, default=4, help="Max repeat factor for class-balanced list")
    parser.add_argument(
        "--balance-strength",
        type=float,
        default=0.45,
        help="0~1. Higher means stronger upsampling for tail classes",
    )

    parser.add_argument("--hardneg-category-id", type=int, default=1)
    parser.add_argument("--hardneg-score-thr", type=float, default=0.25)
    parser.add_argument("--hardneg-iou-bg-thr", type=float, default=0.1)
    parser.add_argument("--hardneg-max-samples", type=int, default=3000)
    parser.add_argument("--hardneg-repeat", type=int, default=2)

    parser.add_argument("--export-conf", type=float, default=0.001)
    parser.add_argument("--export-iou", type=float, default=0.7)
    parser.add_argument("--export-max-det", type=int, default=300)

    parser.add_argument("--project", type=Path, default=Path("runs/GAIIC2024_aggressive"))
    parser.add_argument("--name", type=str, default="balanced-hardneg-kfold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestamp", action="store_true", help="Append timestamp to run directory name")
    parser.add_argument("--dry-run", action="store_true", help="Prepare splits/configs only; skip training/mining")

    return parser.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_train_json(args: argparse.Namespace) -> Path:
    if args.train_json is not None:
        return args.train_json.resolve()
    return (args.dataset_root / "train" / "train.json").resolve()


def resolve_train_rgb_path(dataset_root: Path, file_name: str) -> Path:
    p = Path(file_name)
    candidates = [
        (dataset_root / p),
        (dataset_root / "train" / "rgb" / p),
        (dataset_root / "train" / p),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    # Fallback keeps filename stem stable even when file_name contains extra prefixes.
    return (dataset_root / "train" / "rgb" / p.name).resolve()


def resolve_file_from_csv(dataset_root: Path, file_name: str) -> Path:
    p = Path(file_name)
    candidates = [
        p,
        dataset_root / p,
        dataset_root / "train" / "rgb" / p,
        dataset_root / "train" / p,
        dataset_root / "train" / "rgb" / p.name,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (dataset_root / "train" / "rgb" / p.name).resolve()


def load_train_coco(train_json: Path) -> dict:
    with train_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    required_keys = {"images", "annotations", "categories"}
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in train json {train_json}: {missing}")
    return data


def build_image_structures(coco: dict, dataset_root: Path) -> tuple[dict[int, dict], dict[int, list[dict]], dict[int, set[int]], dict[int, Path]]:
    images = {int(im["id"]): im for im in coco["images"]}
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    image_classes: dict[int, set[int]] = defaultdict(set)
    for ann in coco["annotations"]:
        img_id = int(ann["image_id"])
        anns_by_img[img_id].append(ann)
        image_classes[img_id].add(int(ann["category_id"]))

    img_path_by_id = {
        img_id: resolve_train_rgb_path(dataset_root, str(im.get("file_name", f"{img_id:05d}.jpg")))
        for img_id, im in images.items()
    }
    return images, anns_by_img, image_classes, img_path_by_id


def stratified_folds(
    image_ids: list[int],
    image_classes: dict[int, set[int]],
    ann_class_counts: Counter,
    k: int,
    seed: int,
) -> list[list[int]]:
    if k < 2:
        raise ValueError("folds must be >= 2")
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)

    def rarity_score(img_id: int) -> float:
        cls = image_classes.get(img_id, set())
        if not cls:
            return 0.0
        return sum(1.0 / max(1, ann_class_counts[c]) for c in cls)

    ids.sort(key=rarity_score, reverse=True)

    folds: list[list[int]] = [[] for _ in range(k)]
    fold_cls: list[Counter] = [Counter() for _ in range(k)]
    fold_size = [0 for _ in range(k)]

    for img_id in ids:
        cls = image_classes.get(img_id, set())
        best_fold = 0
        best_score = None
        for fi in range(k):
            score = 0.0
            for c in cls:
                score += (fold_cls[fi][c] + 1) / max(1, ann_class_counts[c])
            score += 0.002 * fold_size[fi]
            if best_score is None or score < best_score:
                best_score = score
                best_fold = fi

        folds[best_fold].append(img_id)
        fold_size[best_fold] += 1
        for c in cls:
            fold_cls[best_fold][c] += 1

    return folds


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in lines:
            f.write(x)
            if not x.endswith("\n"):
                f.write("\n")


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def create_val_subset_json(coco: dict, val_ids: set[int], out_path: Path) -> None:
    out = {
        "images": [im for im in coco["images"] if int(im["id"]) in val_ids],
        "annotations": [ann for ann in coco["annotations"] if int(ann["image_id"]) in val_ids],
        "categories": coco["categories"],
    }
    if "info" in coco:
        out["info"] = coco["info"]
    if "licenses" in coco:
        out["licenses"] = coco["licenses"]
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_balanced_train_list(
    train_ids: list[int],
    anns_by_img: dict[int, list[dict]],
    image_classes: dict[int, set[int]],
    img_path_by_id: dict[int, Path],
    max_repeat: int,
    strength: float,
    seed: int,
) -> tuple[list[str], dict[int, int]]:
    cls_counts = Counter()
    for img_id in train_ids:
        for ann in anns_by_img.get(img_id, []):
            cls_counts[int(ann["category_id"])] += 1

    max_count = max(cls_counts.values()) if cls_counts else 1
    class_weight = {c: (max_count / max(1, n)) for c, n in cls_counts.items()}

    lines: list[str] = []
    repeat_hist = Counter()
    for img_id in train_ids:
        cls = image_classes.get(img_id, set())
        if cls:
            max_w = max(class_weight.get(c, 1.0) for c in cls)
            repeat = int(math.ceil(1.0 + (max_w - 1.0) * strength))
        else:
            repeat = 1
        repeat = max(1, min(max_repeat, repeat))
        repeat_hist[repeat] += 1
        p = str(img_path_by_id[img_id])
        for _ in range(repeat):
            lines.append(p)

    rng = random.Random(seed)
    rng.shuffle(lines)
    return lines, dict(repeat_hist)


def append_hardneg_samples(
    dataset_root: Path,
    base_train_list: Path,
    hardneg_csv: Path,
    out_list: Path,
    repeat: int,
    seed: int,
) -> tuple[int, int]:
    base_lines = [x.strip() for x in base_train_list.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not hardneg_csv.exists():
        write_lines(out_list, base_lines)
        return len(base_lines), 0

    names = []
    with hardneg_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            fn = row.get("file_name", "")
            if fn:
                names.append(fn)

    unique_paths = sorted({str(resolve_file_from_csv(dataset_root, x)) for x in names})
    extra = []
    for p in unique_paths:
        for _ in range(max(1, repeat)):
            extra.append(p)

    merged = base_lines + extra
    rng = random.Random(seed)
    rng.shuffle(merged)
    write_lines(out_list, merged)
    return len(merged), len(unique_paths)


def run_subprocess(cmd: list[str]) -> None:
    log("[CMD] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def export_predictions_for_list(
    weights: Path,
    list_file: Path,
    dataset_root: Path,
    names: list[str],
    out_json: Path,
    args: argparse.Namespace,
) -> None:
    temp_yaml = out_json.parent / (out_json.stem + "_export.yaml")
    tmp_project = out_json.parent / "_tmp_export"
    tmp_name = out_json.stem
    tmp_dir = tmp_project / tmp_name

    data = {
        "path": str(dataset_root.resolve()),
        "train": "train/rgb",
        "val": str(list_file.resolve()),
        "nc": len(names),
        "names": names,
    }
    write_yaml(temp_yaml, data)

    try:
        model = YOLO(str(weights))
        model.val(
            data=str(temp_yaml),
            split="val",
            imgsz=args.imgsz,
            conf=args.export_conf,
            iou=args.export_iou,
            max_det=args.export_max_det,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            use_simotm=args.use_simotm,
            channels=args.channels,
            pairs_rgb_ir=[args.pairs_rgb, args.pairs_ir],
            save_json=True,
            plots=False,
            save=False,
            verbose=False,
            project=str(tmp_project),
            name=tmp_name,
            exist_ok=True,
        )
        pred_json = tmp_dir / "predictions.json"
        if not pred_json.exists():
            raise FileNotFoundError(f"predictions.json not found: {pred_json}")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pred_json, out_json)
    finally:
        temp_yaml.unlink(missing_ok=True)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if tmp_project.exists() and not any(tmp_project.iterdir()):
            tmp_project.rmdir()


def run_training_round(
    model_weights: Path,
    fold_data_yaml: Path,
    run_project: Path,
    run_name: str,
    epochs: int,
    args: argparse.Namespace,
) -> Path:
    model = YOLO(str(model_weights))
    model.train(
        data=str(fold_data_yaml),
        epochs=epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        project=str(run_project),
        name=run_name,
        exist_ok=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        patience=args.patience,
        use_simotm=args.use_simotm,
        channels=args.channels,
        pairs_rgb_ir=[args.pairs_rgb, args.pairs_ir],
    )

    run_dir = run_project / run_name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    if best_pt.exists():
        return best_pt
    if last_pt.exists():
        return last_pt
    raise FileNotFoundError(f"No checkpoint found under {run_dir}")


def parse_best_map(results_csv: Path) -> tuple[float, int]:
    if not results_csv.exists():
        return float("nan"), -1

    best_map = float("-inf")
    best_epoch = -1
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(float(row.get("epoch", -1)))
            v = None
            for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95(M)"):
                if key in row and row[key] not in (None, ""):
                    v = float(row[key])
                    break
            if v is None:
                continue
            if v > best_map:
                best_map = v
                best_epoch = epoch

    if best_epoch < 0:
        return float("nan"), -1
    return best_map, best_epoch


def create_fold_data_yaml(
    dataset_root: Path,
    train_list: Path,
    val_list: Path,
    names: list[str],
    out_yaml: Path,
) -> None:
    data = {
        "path": str(dataset_root.resolve()),
        "train": str(train_list.resolve()),
        "val": str(val_list.resolve()),
        "nc": len(names),
        "names": names,
    }
    write_yaml(out_yaml, data)


def run_hardneg_mining(
    dataset_root: Path,
    train_json: Path,
    pred_json: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Path:
    cmd = [
        sys.executable,
        "tools/mine_car_hardnegatives.py",
        "--gt",
        str(train_json),
        "--dt",
        str(pred_json),
        "--image-root",
        str((dataset_root / "train" / "rgb").resolve()),
        "--out-dir",
        str(out_dir.resolve()),
        "--category-id",
        str(args.hardneg_category_id),
        "--score-thr",
        str(args.hardneg_score_thr),
        "--iou-bg-thr",
        str(args.hardneg_iou_bg_thr),
        "--max-samples",
        str(args.hardneg_max_samples),
        "--pairs-rgb",
        args.pairs_rgb,
        "--pairs-ir",
        args.pairs_ir,
        "--no-crops",
    ]
    run_subprocess(cmd)
    return out_dir / "hardnegatives.csv"


def maybe_timestamp_name(base: str, enable: bool) -> str:
    if not enable:
        return base
    return f"{base}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    train_json = resolve_train_json(args)
    init_weights = args.initial_weights.resolve()

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset-root not found: {args.dataset_root}")
    if not train_json.exists():
        raise FileNotFoundError(f"train-json not found: {train_json}")
    if not init_weights.exists() and not args.dry_run:
        raise FileNotFoundError(f"initial-weights not found: {init_weights}")
    if args.rounds < 1:
        raise ValueError("rounds must be >= 1")

    run_name = maybe_timestamp_name(args.name, args.timestamp)
    run_root = (args.project / run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    log(f"[INFO] run_root={run_root}")
    log(f"[INFO] dataset_root={args.dataset_root}")
    log(f"[INFO] train_json={train_json}")
    log(f"[INFO] initial_weights={init_weights}")

    coco = load_train_coco(train_json)
    images, anns_by_img, image_classes, img_path_by_id = build_image_structures(coco, args.dataset_root)

    categories = sorted(coco["categories"], key=lambda x: int(x["id"]))
    cat_names = [str(x["name"]) for x in categories]
    image_ids = sorted(images.keys())

    ann_class_counts = Counter(int(ann["category_id"]) for ann in coco["annotations"])
    folds = stratified_folds(image_ids, image_classes, ann_class_counts, args.folds, args.seed)

    fold_records: list[dict] = []
    training_root = run_root / "kfold_training"
    training_root.mkdir(parents=True, exist_ok=True)

    for fold_idx, val_ids in enumerate(folds):
        fold_name = f"fold{fold_idx}"
        fold_dir = run_root / "folds" / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_set = set(val_ids)
        train_ids = [x for x in image_ids if x not in val_set]

        val_list = [str(img_path_by_id[x]) for x in val_ids]
        val_list_file = fold_dir / "val_images.txt"
        write_lines(val_list_file, val_list)

        train_balanced_lines, repeat_hist = build_balanced_train_list(
            train_ids=train_ids,
            anns_by_img=anns_by_img,
            image_classes=image_classes,
            img_path_by_id=img_path_by_id,
            max_repeat=args.balance_max_repeat,
            strength=args.balance_strength,
            seed=args.seed + fold_idx,
        )
        train_list_file = fold_dir / "train_balanced_round0.txt"
        write_lines(train_list_file, train_balanced_lines)

        val_gt_json = fold_dir / "val_gt.json"
        create_val_subset_json(coco, val_set, val_gt_json)

        fold_meta = {
            "fold": fold_idx,
            "train_images": len(train_ids),
            "val_images": len(val_ids),
            "train_list_lines_round0": len(train_balanced_lines),
            "repeat_hist_round0": repeat_hist,
            "val_gt_json": str(val_gt_json),
        }
        (fold_dir / "fold_meta.json").write_text(json.dumps(fold_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        current_weights = init_weights
        current_train_list = train_list_file
        round_records: list[dict] = []

        for round_idx in range(args.rounds):
            round_epochs = args.base_epochs if round_idx == 0 else args.online_epochs
            fold_yaml = fold_dir / f"fold_data_round{round_idx}.yaml"
            create_fold_data_yaml(
                dataset_root=args.dataset_root,
                train_list=current_train_list,
                val_list=val_list_file,
                names=cat_names,
                out_yaml=fold_yaml,
            )

            run_name_round = f"{run_name}-{fold_name}-r{round_idx}"
            run_dir_round = training_root / run_name_round

            log(
                f"[FOLD {fold_idx}][ROUND {round_idx}] "
                f"epochs={round_epochs} train_list={current_train_list}"
            )

            if args.dry_run:
                trained_weights = current_weights
                best_map = float("nan")
                best_epoch = -1
            else:
                trained_weights = run_training_round(
                    model_weights=current_weights,
                    fold_data_yaml=fold_yaml,
                    run_project=training_root,
                    run_name=run_name_round,
                    epochs=round_epochs,
                    args=args,
                )
                best_map, best_epoch = parse_best_map(run_dir_round / "results.csv")

            rec = {
                "fold": fold_idx,
                "round": round_idx,
                "epochs": round_epochs,
                "train_list": str(current_train_list),
                "fold_yaml": str(fold_yaml),
                "run_dir": str(run_dir_round),
                "weights": str(trained_weights),
                "best_map50_95": best_map,
                "best_epoch": best_epoch,
            }
            round_records.append(rec)

            if round_idx < args.rounds - 1:
                if args.dry_run:
                    next_train_list = fold_dir / f"train_balanced_hardneg_round{round_idx + 1}.txt"
                    copied_lines = [x.strip() for x in current_train_list.read_text(encoding="utf-8").splitlines() if x.strip()]
                    write_lines(next_train_list, copied_lines)
                    merged_count, hardneg_unique = len(copied_lines), 0
                else:
                    train_pred_json = fold_dir / f"train_pred_round{round_idx}.json"
                    export_predictions_for_list(
                        weights=trained_weights,
                        list_file=current_train_list,
                        dataset_root=args.dataset_root,
                        names=cat_names,
                        out_json=train_pred_json,
                        args=args,
                    )

                    hardneg_dir = fold_dir / f"hardneg_round{round_idx}"
                    hardneg_csv = run_hardneg_mining(
                        dataset_root=args.dataset_root,
                        train_json=train_json,
                        pred_json=train_pred_json,
                        out_dir=hardneg_dir,
                        args=args,
                    )
                    next_train_list = fold_dir / f"train_balanced_hardneg_round{round_idx + 1}.txt"
                    merged_count, hardneg_unique = append_hardneg_samples(
                        dataset_root=args.dataset_root,
                        base_train_list=current_train_list,
                        hardneg_csv=hardneg_csv,
                        out_list=next_train_list,
                        repeat=args.hardneg_repeat,
                        seed=args.seed + fold_idx * 100 + round_idx,
                    )

                rec["next_train_list"] = str(next_train_list)
                rec["next_train_list_lines"] = merged_count
                rec["hardneg_unique_images"] = hardneg_unique
                current_train_list = next_train_list

            current_weights = trained_weights

        valid_rounds = [r for r in round_records if not math.isnan(r["best_map50_95"]) ]
        if valid_rounds:
            best_round = max(valid_rounds, key=lambda x: x["best_map50_95"])
        else:
            best_round = round_records[-1]

        fold_summary = {
            "fold": fold_idx,
            "best_round": best_round["round"],
            "best_map50_95": best_round["best_map50_95"],
            "best_epoch": best_round["best_epoch"],
            "best_weights": best_round["weights"],
            "train_images": len(train_ids),
            "val_images": len(val_ids),
            "round_records": round_records,
        }
        (fold_dir / "fold_round_summary.json").write_text(
            json.dumps(fold_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        fold_records.append(fold_summary)

    fold_records_sorted = sorted(
        fold_records,
        key=lambda x: x["best_map50_95"] if not math.isnan(x["best_map50_95"]) else -1.0,
        reverse=True,
    )

    leaderboard_csv = run_root / "kfold_leaderboard.csv"
    with leaderboard_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "fold", "best_round", "best_map50_95", "best_epoch", "best_weights", "train_images", "val_images"])
        for rank, rec in enumerate(fold_records_sorted, start=1):
            m = rec["best_map50_95"]
            w.writerow(
                [
                    rank,
                    rec["fold"],
                    rec["best_round"],
                    f"{m:.6f}" if not math.isnan(m) else "nan",
                    rec["best_epoch"],
                    rec["best_weights"],
                    rec["train_images"],
                    rec["val_images"],
                ]
            )

    all_summary = {
        "run_root": str(run_root),
        "seed": args.seed,
        "folds": args.folds,
        "rounds": args.rounds,
        "initial_weights": str(init_weights),
        "records": fold_records_sorted,
    }
    (run_root / "kfold_summary.json").write_text(json.dumps(all_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    best_global = fold_records_sorted[0] if fold_records_sorted else None
    best_global_path = run_root / "best_fold.json"
    best_global_path.write_text(json.dumps(best_global, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log(f"[DONE] leaderboard: {leaderboard_csv}")
    log(f"[DONE] summary: {run_root / 'kfold_summary.json'}")
    if best_global:
        log(
            "[DONE] best_fold "
            f"fold={best_global['fold']} round={best_global['best_round']} "
            f"map50_95={best_global['best_map50_95']:.6f}"
        )


if __name__ == "__main__":
    main()
