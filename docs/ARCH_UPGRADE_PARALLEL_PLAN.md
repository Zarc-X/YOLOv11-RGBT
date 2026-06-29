# 架构升级并行执行清单（单卡版）

本清单对应脚本：
- [tools/run_arch_upgrade_parallel.sh](tools/run_arch_upgrade_parallel.sh)

目标：
- Phase1：并行筛选 3 条架构分支（E2/E3/E4）
- Phase2：自动选 TopK（默认2条）进入 E5/E6 并行续训
- 全程保留止损规则，减少无效训练

## 0. 默认策略

- E2: DeepDBB（896）
- E3: EFM（768）
- E4: NiNfusion（896）
- seed: MCF-final best
- phase1: pilot=20, full=100
- phase2: E5(3 folds, 1 round), E6(30 epochs)

默认脚本中 E6 分辨率设置为 `768`，避免此前 `640` 导致的回退风险。

## 1. 一键全流程（推荐）

```bash
cd /root/autodl-tmp/YOLOv11-RGBT
bash tools/run_arch_upgrade_parallel.sh
```

等价于 `MODE=all`：先 phase1 后 phase2。

## 2. 分阶段运行

只跑 phase1：

```bash
cd /root/autodl-tmp/YOLOv11-RGBT
MODE=phase1 bash tools/run_arch_upgrade_parallel.sh
```

只跑 phase2（自动读取最近一次 run）：

```bash
cd /root/autodl-tmp/YOLOv11-RGBT
MODE=phase2 bash tools/run_arch_upgrade_parallel.sh
```

查看状态与日志：

```bash
cd /root/autodl-tmp/YOLOv11-RGBT
MODE=status bash tools/run_arch_upgrade_parallel.sh
```

## 3. 常用覆盖参数

### 3.1 指定 run id（续跑/复盘）

```bash
cd /root/autodl-tmp/YOLOv11-RGBT
RUN_ID=20260628-xxxxxx MODE=status bash tools/run_arch_upgrade_parallel.sh
```

### 3.2 调整 TopK

```bash
cd /root/autodl-tmp/YOLOv11-RGBT
TOPK=2 MODE=phase2 bash tools/run_arch_upgrade_parallel.sh
```

### 3.3 让 E6 分辨率跟随各分支

```bash
cd /root/autodl-tmp/YOLOv11-RGBT
E6_IMGSZ=match MODE=phase2 bash tools/run_arch_upgrade_parallel.sh
```

### 3.4 开启更严格止损

```bash
cd /root/autodl-tmp/YOLOv11-RGBT
EARLY_DROP_GAP=0.004 MIN_GAIN=0.003 MODE=phase1 bash tools/run_arch_upgrade_parallel.sh
```

### 3.5 同 RUN_ID 重启保护

默认会阻止同一 RUN_ID 下重复启动 phase1/phase2 任务，防止进程叠加。

若你明确要强制重启（不推荐），可使用：

```bash
cd /root/autodl-tmp/YOLOv11-RGBT
FORCE_RELAUNCH=1 MODE=phase1 RUN_ID=<你的RUN_ID> bash tools/run_arch_upgrade_parallel.sh
```

## 4. 关键产物

每次运行主目录：
- `runs/GAIIC2024_arch_upgrade_parallel/<RUN_ID>/`

重点文件：
- `phase1_leaderboard.csv`：三分支比较表
- `phase1_topk.json`：进入 phase2 的分支列表
- `phase2_launch_manifest.csv`：phase2 任务启动记录
- `logs/*.log`：各任务日志

## 5. 与现有脚本关系

该并行脚本内部复用了现有流程：
- [tools/run_e1_to_e6_pipeline.py](tools/run_e1_to_e6_pipeline.py)
- [tools/run_e5_e6_from_weight.py](tools/run_e5_e6_from_weight.py)

因此你可以继续保持原来的训练与提交链路，不需要改历史脚本。
