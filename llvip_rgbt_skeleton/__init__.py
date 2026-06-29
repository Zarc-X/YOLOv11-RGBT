"""Minimal PyTorch skeleton for LLVIP-style RGB-TIR detection."""

from .dataset import GAIICRGBTDataset, build_dataloader
from .model import RGBTDetector, DetectionLoss
