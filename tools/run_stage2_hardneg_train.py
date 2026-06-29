#!/usr/bin/env python3
from ultralytics import YOLO


def main() -> None:
    weights = "runs/GAIIC2024_archscan/archscan-RGBT-midfusion/weights/best.pt"
    data = "runs/analysis/stage2_hardneg_midfusion/GAIIC2024-rgbt-stage2-hardneg.yaml"

    model = YOLO(weights)
    model.train(
        data=data,
        epochs=30,
        batch=16,
        imgsz=640,
        workers=2,
        device="0",
        project="runs/GAIIC2024_stage2",
        name="stage2-midfusion-hardneg",
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
        use_simotm="RGBT",
        channels=4,
        pairs_rgb_ir=["rgb", "tir"],
    )


if __name__ == "__main__":
    main()
