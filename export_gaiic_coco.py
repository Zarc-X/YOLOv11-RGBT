#!/usr/bin/env python3
import argparse
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export COCO detection json for GAIIC val/test using YOLOv11-RGBT")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights (.pt), e.g. runs/GAIIC2024/.../weights/best.pt",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        required=True,
        help="Dataset split to infer",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/root/autodl-tmp/GAIIC2024-赛道1-目标检测任务",
        help="GAIIC dataset root directory",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable test-time augmentation for inference/export.",
    )
    parser.add_argument(
        "--use-simotm",
        type=str,
        default="",
        help="Override inference mode (e.g. RGB, RGBT, RGBRGB6C). Empty means auto from run args.yaml",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=0,
        help="Override channel count. 0 means auto from run args.yaml",
    )
    parser.add_argument(
        "--pairs-rgb",
        type=str,
        default="",
        help="Override RGB folder token for paired mode. Empty means auto from run args.yaml",
    )
    parser.add_argument(
        "--pairs-ir",
        type=str,
        default="",
        help="Override IR folder token for paired mode. Empty means auto from run args.yaml",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output json path. If empty: val->val_results_coco_tmp.json, test->test_results_coco.json under run dir",
    )
    parser.add_argument(
        "--export-engine",
        type=str,
        choices=["val", "predict"],
        default="val",
        help="Export backend. 'val' reuses Ultralytics val pipeline (closer to leaderboard behavior); 'predict' uses direct model.predict().",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for val-engine export.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Inference device, e.g. 0 or cpu.",
    )
    return parser.parse_args()


def load_run_args(weights_path: Path):
    run_args_path = weights_path.resolve().parent.parent / "args.yaml"
    if not run_args_path.exists():
        return {}
    try:
        with open(run_args_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[WARN] Failed to parse run args {run_args_path}: {e}")
        return {}


def resolve_infer_config(args, run_args):
    use_simotm = args.use_simotm or run_args.get("use_simotm") or "RGBT"
    channels = args.channels if args.channels > 0 else int(run_args.get("channels") or 4)

    pairs = run_args.get("pairs_rgb_ir")
    if not (isinstance(pairs, list) and len(pairs) == 2 and all(isinstance(x, str) for x in pairs)):
        pairs = ["rgb", "tir"]

    pairs_rgb = args.pairs_rgb or pairs[0]
    pairs_ir = args.pairs_ir or pairs[1]

    return use_simotm, channels, pairs_rgb, pairs_ir


def choose_output_path(weights_path: Path, split: str, out_arg: str) -> Path:
    if out_arg:
        return Path(out_arg)
    run_dir = weights_path.resolve().parent.parent
    if split == "val":
        return run_dir / "val_results_coco_tmp.json"
    return run_dir / "test_results_coco.json"


def make_temp_data_yaml(dataset_root: Path, split: str) -> Path:
    """Build a temporary data yaml that points val to the requested split.

    This allows using model.val(save_json=True) for both val and test export so postprocess behavior
    is aligned with validator logic.
    """
    names = ["car", "truck", "bus", "van", "freight_car"]
    data = {
        "path": str(dataset_root),
        "train": "train/rgb",
        "val": f"{split}/rgb",
        "nc": len(names),
        "names": names,
    }
    fd, p = tempfile.mkstemp(prefix=f"gaiic_export_{split}_", suffix=".yaml")
    Path(p).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    Path(p).chmod(0o644)
    return Path(p)


def export_via_val_engine(
    model: YOLO,
    out_json: Path,
    split: str,
    dataset_root: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    use_simotm: str,
    channels: int,
    pairs_rgb: str,
    pairs_ir: str,
    batch: int,
    augment: bool,
    device: str,
):
    temp_yaml = make_temp_data_yaml(dataset_root, split)
    tmp_project = out_json.parent / "_tmp_val_export"
    tmp_name = f"{split}_backend"
    tmp_dir = tmp_project / tmp_name

    try:
        model.val(
            data=str(temp_yaml),
            split="val",
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            augment=augment,
            device=device,
            use_simotm=use_simotm,
            channels=channels,
            pairs_rgb_ir=[pairs_rgb, pairs_ir],
            save_json=True,
            plots=False,
            save=False,
            verbose=False,
            project=str(tmp_project),
            name=tmp_name,
            exist_ok=True,
            batch=batch,
        )

        pred_json = tmp_dir / "predictions.json"
        if not pred_json.exists():
            raise FileNotFoundError(f"Val backend did not produce predictions json: {pred_json}")

        records = json.loads(pred_json.read_text(encoding="utf-8"))
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
    finally:
        if temp_yaml.exists():
            temp_yaml.unlink(missing_ok=True)
        # Keep workspace clean: temporary val-export artifacts are copied out then removed.
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if tmp_project.exists() and not any(tmp_project.iterdir()):
            tmp_project.rmdir()


def export_via_predict_engine(
    model: YOLO,
    source_dir: Path,
    out_json: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    use_simotm: str,
    channels: int,
    pairs_rgb: str,
    pairs_ir: str,
    augment: bool,
    device: str,
):
    records = []
    predict_kwargs = dict(
        source=str(source_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        save=False,
        stream=True,
        verbose=False,
        augment=augment,
        use_simotm=use_simotm,
        channels=channels,
    )

    if use_simotm in {"RGBT", "RGBRGB6C"}:
        predict_kwargs["pairs_rgb_ir"] = [pairs_rgb, pairs_ir]

    for r in model.predict(**predict_kwargs):
        stem = Path(r.path).stem
        image_id = int(stem) if stem.isnumeric() else stem

        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for b, s, c in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = [float(v) for v in b]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue

            records.append(
                {
                    "image_id": image_id,
                    "category_id": int(c) + 1,
                    "bbox": [round(x1, 3), round(y1, 3), round(w, 3), round(h, 3)],
                    "score": round(float(s), 6),
                }
            )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


def quantile(sorted_vals, p):
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, max(0, int(p * len(sorted_vals)) - 1))
    return sorted_vals[i]


def main():
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    dataset_root = Path(args.dataset_root)
    source_dir = dataset_root / args.split / "rgb"
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")

    out_json = choose_output_path(weights, args.split, args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    run_args = load_run_args(weights)
    use_simotm, channels, pairs_rgb, pairs_ir = resolve_infer_config(args, run_args)

    print(f"infer_use_simotm {use_simotm}")
    print(f"infer_channels {channels}")
    print(f"infer_pairs {pairs_rgb},{pairs_ir}")
    print(f"export_engine {args.export_engine}")
    print(f"augment {args.augment}")
    print(f"device {args.device}")

    model = YOLO(str(weights))
    if args.export_engine == "val":
        export_via_val_engine(
            model=model,
            out_json=out_json,
            split=args.split,
            dataset_root=dataset_root,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            use_simotm=use_simotm,
            channels=channels,
            pairs_rgb=pairs_rgb,
            pairs_ir=pairs_ir,
            batch=args.batch,
            augment=args.augment,
            device=args.device,
        )
    else:
        export_via_predict_engine(
            model=model,
            source_dir=source_dir,
            out_json=out_json,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            use_simotm=use_simotm,
            channels=channels,
            pairs_rgb=pairs_rgb,
            pairs_ir=pairs_ir,
            augment=args.augment,
            device=args.device,
        )

    with open(out_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    by = defaultdict(int)
    for x in records:
        by[x["image_id"]] += 1
    vals = sorted(by.values())

    print(f"out_json {out_json}")
    print(f"pred_records {len(records)}")
    print(f"images_with_pred {len(by)}")
    if vals:
        print(f"det_per_image_mean {round(sum(vals) / len(vals), 4)}")
        print(f"det_per_image_p50 {quantile(vals, 0.5)}")
        print(f"det_per_image_p90 {quantile(vals, 0.9)}")
        print(f"det_per_image_p95 {quantile(vals, 0.95)}")
        print(f"det_per_image_p99 {quantile(vals, 0.99)}")


if __name__ == "__main__":
    main()
