from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class TinyBackbone(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            _conv_block(in_channels, 32, stride=2),
            _conv_block(32, 64, stride=2),
            _conv_block(64, 128, stride=2),
        )
        self.refine = nn.Sequential(
            _conv_block(128, 128),
            _conv_block(128, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.refine(x)


class FusionBlock(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, rgb_feats: torch.Tensor, tir_feats: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([rgb_feats, tir_feats], dim=1))


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pred = nn.Sequential(
            _conv_block(in_channels, in_channels),
            nn.Conv2d(in_channels, 5 + num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred(x)


class RGBTDetector(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.rgb_backbone = TinyBackbone(in_channels=3)
        self.tir_backbone = TinyBackbone(in_channels=1)
        self.fusion = FusionBlock(channels=128)
        self.head = DetectionHead(in_channels=128, num_classes=num_classes)

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor) -> torch.Tensor:
        rgb_feats = self.rgb_backbone(rgb)
        tir_feats = self.tir_backbone(tir)
        fused_feats = self.fusion(rgb_feats, tir_feats)
        return self.head(fused_feats)


@dataclass
class LossOutput:
    total: torch.Tensor
    box: torch.Tensor
    obj: torch.Tensor
    cls: torch.Tensor


class DetectionLoss(nn.Module):
    def __init__(self, num_classes: int, box_weight: float = 5.0, obj_weight: float = 1.0, cls_weight: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    def _build_targets(self, targets: List[torch.Tensor], pred_shape: Tuple[int, int, int, int], device: torch.device):
        batch_size, _, grid_h, grid_w = pred_shape
        obj_target = torch.zeros((batch_size, 1, grid_h, grid_w), device=device)
        cls_target = torch.zeros((batch_size, self.num_classes, grid_h, grid_w), device=device)
        box_target = torch.zeros((batch_size, 4, grid_h, grid_w), device=device)

        for batch_index, target in enumerate(targets):
            if target.numel() == 0:
                continue
            for row in target.to(device):
                class_id = int(row[0].item())
                xc, yc, width, height = row[1:].tolist()
                grid_x = min(grid_w - 1, max(0, int(xc * grid_w)))
                grid_y = min(grid_h - 1, max(0, int(yc * grid_h)))

                obj_target[batch_index, 0, grid_y, grid_x] = 1.0
                cls_target[batch_index, class_id, grid_y, grid_x] = 1.0
                box_target[batch_index, 0, grid_y, grid_x] = xc * grid_w - grid_x
                box_target[batch_index, 1, grid_y, grid_x] = yc * grid_h - grid_y
                box_target[batch_index, 2, grid_y, grid_x] = width
                box_target[batch_index, 3, grid_y, grid_x] = height

        return obj_target, cls_target, box_target

    def forward(self, preds: torch.Tensor, targets: List[torch.Tensor]) -> LossOutput:
        box_raw, obj_raw, cls_raw = preds[:, :4], preds[:, 4:5], preds[:, 5:]
        box_pred = torch.sigmoid(box_raw)

        obj_target, cls_target, box_target = self._build_targets(targets, preds.shape, preds.device)

        obj_loss = self.bce(obj_raw, obj_target)
        cls_loss = self.bce(cls_raw, cls_target)

        positive_mask = obj_target.expand_as(box_pred) > 0.5
        if positive_mask.any():
            box_loss = self.l1(box_pred[positive_mask], box_target[positive_mask])
        else:
            box_loss = box_pred.sum() * 0.0

        total = self.box_weight * box_loss + self.obj_weight * obj_loss + self.cls_weight * cls_loss
        return LossOutput(total=total, box=box_loss, obj=obj_loss, cls=cls_loss)
