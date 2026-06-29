import argparse
import warnings
from pathlib import Path

from ultralytics import YOLO


warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv11-RGBT on the GAIIC2024 dataset')
    parser.add_argument('--model', type=str, default='ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml')
    parser.add_argument('--data', type=str, default='ultralytics/cfg/datasets/GAIIC2024-rgbt.yaml')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--project', type=str, default='runs/GAIIC2024')
    parser.add_argument('--name', type=str, default='GAIIC2024-yolo11n-RGBT-midfusion')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--pretrained-weights', type=str, default='')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--close-mosaic', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = YOLO(args.model)
    if args.pretrained_weights:
        weights_path = Path(args.pretrained_weights)
        if not weights_path.exists():
            raise FileNotFoundError(f'Pretrained weights not found: {weights_path}')
        model.load(str(weights_path))

    model.train(
        data=args.data,
        cache=False,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        close_mosaic=args.close_mosaic,
        workers=args.workers,
        device=args.device,
        optimizer=args.optimizer,
        use_simotm='RGBT',
        channels=4,
        pairs_rgb_ir=['rgb', 'tir'],
        project=args.project,
        name=args.name,
        resume=args.resume or False,
    )