from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass
class SampleRecord:
    rgb_path: Path
    tir_path: Path
    label_path: Path
    image_id: str


def _load_label_file(label_path: Path) -> np.ndarray:
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)

    rows: List[List[float]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = float(parts[0])
        xc = float(parts[1])
        yc = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        rows.append([cls_id, xc, yc, width, height])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _resize_image(image: Image.Image, size: int, mode: str) -> Image.Image:
    if image.mode != mode:
        image = image.convert(mode)
    return image.resize((size, size), Image.BILINEAR)


class GAIICRGBTDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_size: int = 640,
        rgb_dir: str = "rgb",
        tir_dir: str = "tir",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.rgb_dir = self.root / split / rgb_dir
        self.tir_dir = self.root / split / tir_dir

        if not self.rgb_dir.exists():
            raise FileNotFoundError(f"Missing RGB directory: {self.rgb_dir}")
        if not self.tir_dir.exists():
            raise FileNotFoundError(f"Missing TIR directory: {self.tir_dir}")

        self.records: List[SampleRecord] = []
        for rgb_path in sorted(self.rgb_dir.glob("*")):
            if rgb_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            tir_path = self.tir_dir / rgb_path.name
            label_path = self.rgb_dir / f"{rgb_path.stem}.txt"
            if not tir_path.exists():
                continue
            self.records.append(
                SampleRecord(
                    rgb_path=rgb_path,
                    tir_path=tir_path,
                    label_path=label_path,
                    image_id=rgb_path.stem,
                )
            )

        if not self.records:
            raise RuntimeError(f"No paired RGB/TIR samples found under {self.rgb_dir} and {self.tir_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def _load_pair(self, record: SampleRecord) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_image = _resize_image(Image.open(record.rgb_path), self.img_size, "RGB")
        tir_image = _resize_image(Image.open(record.tir_path), self.img_size, "L")

        rgb_array = np.asarray(rgb_image, dtype=np.float32) / 255.0
        tir_array = np.asarray(tir_image, dtype=np.float32) / 255.0

        rgb_tensor = torch.from_numpy(np.transpose(rgb_array, (2, 0, 1))).contiguous()
        tir_tensor = torch.from_numpy(tir_array[None, :, :]).contiguous()
        return rgb_tensor, tir_tensor

    def __getitem__(self, index: int):
        record = self.records[index]
        rgb_tensor, tir_tensor = self._load_pair(record)
        labels = _load_label_file(record.label_path)

        target = torch.from_numpy(labels.copy()) if len(labels) else torch.zeros((0, 5), dtype=torch.float32)
        sample = {
            "rgb": rgb_tensor,
            "tir": tir_tensor,
            "targets": target,
            "image_id": record.image_id,
            "rgb_path": str(record.rgb_path),
            "tir_path": str(record.tir_path),
        }
        return sample


def collate_batch(batch: Sequence[dict]) -> dict:
    rgb = torch.stack([item["rgb"] for item in batch], dim=0)
    tir = torch.stack([item["tir"] for item in batch], dim=0)
    targets = [item["targets"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    rgb_paths = [item["rgb_path"] for item in batch]
    tir_paths = [item["tir_path"] for item in batch]
    return {
        "rgb": rgb,
        "tir": tir,
        "targets": targets,
        "image_ids": image_ids,
        "rgb_paths": rgb_paths,
        "tir_paths": tir_paths,
    }


def build_dataloader(
    root: str | Path,
    split: str,
    img_size: int,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = False,
) -> DataLoader:
    dataset = GAIICRGBTDataset(root=root, split=split, img_size=img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
        drop_last=shuffle,
    )
