from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from .dataset import build_dataloader
from .model import DetectionLoss, RGBTDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal PyTorch RGBT skeleton on GAIIC2024")
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/GAIIC2024-赛道1-目标检测任务"))
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=Path, default=Path("runs/llvip_rgbt_skeleton"))
    return parser.parse_args()


def train_one_epoch(model: nn.Module, criterion: DetectionLoss, loader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        rgb = batch["rgb"].to(device)
        tir = batch["tir"].to(device)
        targets = batch["targets"]

        preds = model(rgb, tir)
        loss_out = criterion(preds, targets)

        optimizer.zero_grad(set_to_none=True)
        loss_out.total.backward()
        optimizer.step()

        total_loss += float(loss_out.total.detach().cpu())
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(model: nn.Module, criterion: DetectionLoss, loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        rgb = batch["rgb"].to(device)
        tir = batch["tir"].to(device)
        targets = batch["targets"]
        preds = model(rgb, tir)
        loss_out = criterion(preds, targets)
        total_loss += float(loss_out.total.detach().cpu())
        num_batches += 1
    return total_loss / max(1, num_batches)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(args.data_root, "train", args.img_size, args.batch_size, args.workers, shuffle=True)
    val_loader = build_dataloader(args.data_root, "val", args.img_size, args.batch_size, args.workers, shuffle=False)

    model = RGBTDetector(num_classes=5).to(device)
    criterion = DetectionLoss(num_classes=5)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device)
        val_loss = evaluate(model, criterion, val_loader, device)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(ckpt, args.save_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, args.save_dir / "best.pt")

        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
