# GAIIC2024 RGBT PyTorch 重构骨架

这份骨架面向“把 LLVIP 里 TensorFlow 的融合模块重构成 PyTorch，然后用于 GAIIC2024 赛道1的 RGBT 目标检测”的场景。当前这份 GAIIC 数据是 2D 检测数据，不是点云任务，所以训练目标应该是 RGB+TIR 双分支输入、统一检测头输出。

## 先决条件

1. 训练标签已经从 COCO JSON 转成 YOLO txt，并且 RGB/TIR 两侧都能找到同名标签文件。
2. 类别顺序固定为 `car, truck, bus, van, freight_car`，对应 COCO category id `1..5`。
3. RGB 与 TIR 图像一一配对，文件名保持一致。
4. 训练入口支持 `use_simotm=RGBT`、`channels=4`、`pairs_rgb_ir=['rgb', 'tir']`。

## 建议保留的 PyTorch 模块边界

### 1. 数据适配层

职责：把 GAIIC 的 `train.json / val.json` 转成训练时可直接读取的图片+标签样本。

建议接口：

```python
class GAIICRGBTDataset:
    def __init__(self, root, split, rgb_dir='rgb', tir_dir='tir'):
        ...

    def __getitem__(self, idx):
        # return rgb, tir, targets, meta
        ...
```

### 2. 双分支编码器

职责：分别提取 RGB 与 TIR 特征，前期可共享部分结构，后期再分支。

建议拆分成：

```python
class RGBBackbone(nn.Module):
    ...

class TIRBackbone(nn.Module):
    ...
```

如果你想尽量复用 LLVIP 的 TensorFlow 逻辑，优先迁移这些部分：

1. 输入归一化与对齐逻辑。
2. 特征提取主干。
3. 融合前的投影层。

### 3. 融合模块

职责：在多尺度特征上做跨模态交互。

建议先做最小实现，再逐步替换为更复杂的注意力或门控结构：

```python
class FusionBlock(nn.Module):
    def forward(self, rgb_feats, tir_feats):
        # return fused_feats
        ...
```

最小可训练版建议只保留一条稳定的融合路径：

1. `concat -> 1x1 conv -> norm -> activation`
2. 或 `add -> 1x1 conv -> norm -> activation`

先跑通训练，再加复杂门控。

### 4. 检测头

职责：输出分类和框回归。

建议直接接现成的 YOLO 系列检测头，或者实现一个轻量多尺度头：

```python
class DetectionHead(nn.Module):
    def forward(self, fused_feats):
        # cls_logits, box_regression, objectness
        ...
```

### 5. 损失与后处理

职责：把 GAIIC 的 2D 标注和模型输出对齐。

最小版本只需要：

1. 分类损失
2. 目标置信度损失
3. 框回归损失
4. NMS 后处理

## 最小可训练骨架

```python
class RGBTDetector(nn.Module):
    def __init__(self, rgb_backbone, tir_backbone, fusion, head):
        super().__init__()
        self.rgb_backbone = rgb_backbone
        self.tir_backbone = tir_backbone
        self.fusion = fusion
        self.head = head

    def forward(self, rgb, tir):
        rgb_feats = self.rgb_backbone(rgb)
        tir_feats = self.tir_backbone(tir)
        fused_feats = self.fusion(rgb_feats, tir_feats)
        return self.head(fused_feats)
```

### 推荐训练顺序

1. 先冻结 TIR 分支，只训 RGB 分支 + 融合层 + 检测头。
2. 再解冻 TIR 分支，做全量微调。
3. 最后再尝试更复杂的跨模态交互模块。

这样做的好处是更容易定位数据、对齐、融合、回归四类问题。

## 针对当前 GAIIC 数据集的落地检查

1. `GAIIC2024-rgbt.yaml` 的 `path/train/val/nc/names` 是否指向真实目录和真实类名。
2. `convert_gaiic_coco_to_yolo.py` 生成的标签是否写到了 RGB 和 TIR 两侧。
3. 训练脚本是否明确传入：`use_simotm='RGBT'`、`channels=4`、`pairs_rgb_ir=['rgb', 'tir']`。
4. 先用 `batch=1` 和少量图片做冒烟训练，确认 dataloader 与 loss 正常。

## 你现在这份 GAIIC 数据的类别顺序

```text
0 car
1 truck
2 bus
3 van
4 freight_car
```

## 最短开训命令

```bash
cd /root/autodl-tmp/YOLOv11-RGBT
python convert_gaiic_coco_to_yolo.py --dataset-root /root/autodl-tmp/GAIIC2024-赛道1-目标检测任务
python train_RGBT.py --data ultralytics/cfg/datasets/GAIIC2024-rgbt.yaml --model ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml --device 0
```

如果你要的是“把 LLVIP 的 TensorFlow 模块逐个改成 PyTorch”，下一步建议优先改：输入适配、融合块、检测头、loss 这四层，不要一开始就追求完全等价复刻。