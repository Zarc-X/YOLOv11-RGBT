# YOLO11-RGBT 轻量化高回报改造说明

## 1. 目标与约束

本次改造目标：

1. 在不切换到超大模型（例如 Co-DETR-SwinL）的前提下，最大化保留可迁移的高回报策略。
2. 优先选择对推理开销影响小或仅影响训练期的策略。
3. 提供可在另一张卡直接运行的脚本化实验与提交流程。

约束：

- 推理部署仍以 YOLO11-RGBT 轻量路线为主。
- 尽量复用当前仓库已有能力（DetectAux、P2 检测头、WBF 提交链路）。

---

## 2. 从“原始 YOLO11-RGBT”到“保留有效优化”的演进

### 2.1 原始主线（本仓库基线）

原始 YOLO11-RGBT 主线是双分支 mid-fusion，典型配置：

- `ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml`
- `train_RGBT.py`

这条路线的优势是轻量、易部署，但在跨模态错位鲁棒性、训练稳定性和后期收敛细节上还有提升空间。

### 2.2 借鉴来源与筛选原则

借鉴来源：

1. 新仓库 `gaiic2024`（代码）
2. 论文 `2407.03872v1.pdf`（方法与消融）

筛选原则（只保留轻量高回报项）：

1. 训练期增益明显、推理期开销低。
2. 可直接映射到当前 YOLO11-RGBT 工程结构。
3. 对单卡复现友好。

---

## 3. 本次落地的高回报策略

### 3.1 模态错位增强（新加）

对应借鉴：论文中“RGB/TIR 注册误差鲁棒性”思路 + 新仓库中的模态差异增强。

实现：

- 在 4 通道输入上新增 `RandomModalShift4C`，可对 RGB 或 TIR 模态随机像素平移。
- 已接入 `v8_transforms()` 训练流水线。

代码位置：

- `ultralytics/data/augment.py`
- `ultralytics/cfg/default.yaml`（新增参数 `modal_shift`, `modal_shift_px`, `modal_shift_mode`）

设计要点：

- 默认 `modal_shift=0.0`，不影响旧实验。
- 可按阶段配置不同 shift 强度（Stage1 强、Stage2 弱）。

### 3.2 两阶段训练日程（新加）

对应借鉴：新仓库两阶段 pipeline（前期强增广，后期回归常规增强）。

实现：

- 新脚本 `tools/run_lightweight_twostage.py`。
- Stage1 使用更强增广（mosaic/shift/几何扰动更高）。
- Stage2 从 Stage1 最优权重继续训练，降低增强强度，强调收敛与定位稳定。

设计要点：

- 支持 `--initial-weights` 进行迁移初始化。
- 自动产出 `pipeline_summary.json`，便于后续自动化提交与追踪。

### 3.3 训练期辅助监督头（新加配置）

对应借鉴：论文强调的辅助监督思想 + 本仓库已支持的 `DetectAux` 能力。

实现：

- 新模型配置：
  - `ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-EFMv2-P2-P5NiN-AUX.yaml`
- 基于 `EFMv2-P2-P5NiN` 加入 `DetectAux`。
- 主分支使用 P2/P3/P4/P5 输出，辅助分支使用融合 backbone 特征做训练期监督。

设计要点：

- `DetectAux` 在本仓库 loss 链路已打通（`v8DetectionLoss` 内部已支持 aux 逻辑）。
- 推理可通过 `switch_to_deploy()` 移除辅助头，避免额外推理负担。

### 3.4 轻量并行实验总控（新加）

实现：

- `tools/run_lightweight_highroi_parallel.sh`

内置实验：

1. `lw-efmv2p2p5nin`：稀疏融合 + P2 头 + 两阶段增强
2. `lw-efmv2p2p5nin-aux`：在 1 基础上加训练期辅助头
3. `lw-deepdbb`：DeepDBB 结构基线 + 两阶段增强

可选顺序或并行启动（默认顺序，单卡更安全）。

### 3.5 轻量提交通路（新加）

实现：

- `tools/run_lightweight_submit_chain.sh`

能力：

- 支持 2~3 个模型权重自动导出 test 预测。
- 可选 TTA 导出。
- 自动 WBF 融合 + shrink 生成提交文件。
- 产出 `submission_manifest.csv`。

---

## 4. 新增/修改文件清单

### 4.1 修改文件

1. `ultralytics/data/augment.py`
   - 新增 `RandomModalShift4C`
   - 接入 `v8_transforms()`

2. `ultralytics/cfg/default.yaml`
   - 新增默认参数：
     - `modal_shift`
     - `modal_shift_px`
     - `modal_shift_mode`

### 4.2 新增文件

1. `ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-EFMv2-P2-P5NiN-AUX.yaml`
2. `tools/run_lightweight_twostage.py`
3. `tools/run_lightweight_highroi_parallel.sh`
4. `tools/run_lightweight_submit_chain.sh`
5. `docs/LIGHTWEIGHT_RGBT_HIGHROI_UPGRADE.md`（本文）

---

## 5. 在另一张卡的推荐运行方式

## 5.1 一键跑高回报实验（推荐）

```bash
cd /root/autodl-tmp/YOLOv11-RGBT

MODE=all \
DEVICE=auto \
BATCH=8 \
WORKERS=2 \
RUN_ID=lightroi-$(date +%Y%m%d-%H%M%S) \
STAGE1_EPOCHS=30 \
STAGE2_EPOCHS=70 \
PARALLEL=0 \
bash tools/run_lightweight_highroi_parallel.sh
```

说明：

- `DEVICE=auto` 会自动选择可见 GPU 的 `0` 号设备；单卡时通常应使用 `0` 而不是 `1`。
- 若你要显式指定设备，单卡建议 `DEVICE=0`。
- `PARALLEL=0` 为单卡稳妥模式；如果你确认显存充足可设为 `1`。

## 5.2 单实验精跑（例如 AUX 版本）

```bash
cd /root/autodl-tmp/YOLOv11-RGBT

python tools/run_lightweight_twostage.py \
  --model ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion-EFMv2-P2-P5NiN-AUX.yaml \
  --data ultralytics/cfg/datasets/GAIIC2024-rgbt.yaml \
  --project runs/GAIIC2024_lightweight_manual \
  --name efmv2p2p5nin_aux_r1 \
  --device auto \
  --batch 8 \
  --workers 2 \
  --imgsz 768 \
  --stage1-epochs 30 \
  --stage2-epochs 70
```

## 5.3 轻量融合提交（2~3 模型）

```bash
cd /root/autodl-tmp/YOLOv11-RGBT

MODEL1_WEIGHTS=runs/GAIIC2024_lightweight_highroi/<RUN_ID>/lw-efmv2p2p5nin-s2/weights/best.pt \
MODEL1_IMGSZ=768 \
MODEL2_WEIGHTS=runs/GAIIC2024_lightweight_highroi/<RUN_ID>/lw-efmv2p2p5nin-aux-s2/weights/best.pt \
MODEL2_IMGSZ=768 \
MODEL3_WEIGHTS=runs/GAIIC2024_lightweight_highroi/<RUN_ID>/lw-deepdbb-s2/weights/best.pt \
MODEL3_IMGSZ=896 \
DEVICE=auto \
BATCH=8 \
EXPORT_TTA=1 \
bash tools/run_lightweight_submit_chain.sh
```

## 5.4 若提示 CUDA 不可用（H800 常见）

先在目标 conda 环境里做快速自检：

```bash
nvidia-smi
python - <<'PY'
import torch
print('torch', torch.__version__)
print('torch.version.cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
print('cuda_count', torch.cuda.device_count())
PY
```

如果 `cuda_available=False`：

1. 确认当前容器是否已挂载 GPU（`nvidia-smi` 应能看到 H800）。
2. 确认当前 conda 环境安装的是 CUDA 版 PyTorch，而不是 CPU 版。
3. 单卡场景设备号通常是 `0`；脚本已支持 `DEVICE=auto` 自动纠偏。

你也可以先用脚本内置预检：

```bash
MODE=status DEVICE=auto bash tools/run_lightweight_highroi_parallel.sh
```

---

## 6. 为什么保留这些策略（而不是直接搬大模型）

1. 论文与仓库共同表明：数据增强与训练策略对成绩提升贡献很大。
2. 训练期辅助监督能提升优化效果，但可在推理期去除，不增加部署成本。
3. 轻量架构的上限通常靠“结构小改 + 训练策略 + 小规模集成”共同抬升。
4. 对你当前多卡/单卡切换场景更友好，迁移成本低。

---

## 7. 风险与回滚建议

1. 若出现训练不稳定：先将 `modal_shift` 降到 `0.1` 或关闭。
2. 若显存不足：优先降 `imgsz`、`batch`，其次减少 `workers`。
3. 若 AUX 版本收敛慢：保持两阶段策略，延长 Stage2 10~20 epoch。
4. 若想回到旧行为：
   - 使用原配置（非 AUX）
   - 保持 `modal_shift=0.0`

---

## 8. 结果产物约定

每个实验会生成：

1. `.../<exp>-s1/weights/best.pt`
2. `.../<exp>-s2/weights/best.pt`
3. `.../<exp_parent>/pipeline_summary.json`

这使得后续自动化筛选和提交链路可以直接消费统一摘要文件。
