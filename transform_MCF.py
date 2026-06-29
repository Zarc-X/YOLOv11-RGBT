import torch
import torch.nn as nn
import re
import datetime
import argparse
from pathlib import Path


def load_model_from_checkpoint(model_path: Path, role: str) -> nn.Module:
    """Load a model module from an Ultralytics checkpoint.

    Depending on save path and trainer state, best.pt may store weights under
    `model` or `ema`. This helper picks the first valid module.
    """
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, nn.Module):
        return checkpoint

    if isinstance(checkpoint, dict):
        model_obj = checkpoint.get("model", None)
        ema_obj = checkpoint.get("ema", None)
        chosen = model_obj if isinstance(model_obj, nn.Module) else ema_obj
        if isinstance(chosen, nn.Module):
            return chosen

        keys = ", ".join(sorted(checkpoint.keys()))
        raise ValueError(
            f"{role} checkpoint has no valid nn.Module in 'model' or 'ema': {model_path}. "
            f"Available keys: [{keys}]"
        )

    raise TypeError(f"Unsupported checkpoint type for {role}: {type(checkpoint)} from {model_path}")


def copy_and_modify_layers(source_model_path, target_model_path, output_model_path):
    source_model_path = Path(source_model_path)
    target_model_path = Path(target_model_path)
    output_model_path = Path(output_model_path)

    if not source_model_path.exists():
        raise FileNotFoundError(f"Source model not found: {source_model_path}")
    if not target_model_path.exists():
        raise FileNotFoundError(f"Target model not found: {target_model_path}")
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载源模型和目标模型（兼容 model/ema 两种保存方式）
    source_model = load_model_from_checkpoint(source_model_path, role="source")
    target_model = load_model_from_checkpoint(target_model_path, role="target")

    # 定义复制的层范围
    copy_ranges = [
        # (源层范围, 目标层范围)
        ((0, 4), (2, 6)),  # model.0-8 复制到 model.2-10
        ((0, 0), (10, 10)),  # model.0-8 复制到 model.12-20
        ((1, 4), (11, 14)),  # model.0-8 复制到 model.12-20
        ((5, 6), (17, 18)),  # model.0-8 复制到 model.12-20
        ((5, 6), (19, 20)),  # model.0-8 复制到 model.12-20
        ((7, 8), (23, 24)),  # model.0-8 复制到 model.12-20
        ((7, 8), (25, 26)),  # model.0-8 复制到 model.12-20

        ((9, 23), (29, 43)),  # model.9-22 复制到 model.24-37
    ]

    # 遍历每个复制范围
    for (src_start, src_end), (tgt_start, tgt_end) in copy_ranges:
        # 获取源模型的层
        source_layers = []
        for name, module in source_model.named_modules():
            if name.startswith('model.') and re.match(r'model\.\d+', name):
                match = re.match(r'model\.(\d+)', name)
                if match:
                    layer_index = int(match.group(1))
                    if src_start <= layer_index <= src_end:
                        source_layers.append((name, module))

        # 获取目标模型的层
        target_layers = []
        for name, module in target_model.named_modules():
            if name.startswith('model.') and re.match(r'model\.\d+', name):
                match = re.match(r'model\.(\d+)', name)
                if match:
                    layer_index = int(match.group(1))
                    if tgt_start <= layer_index <= tgt_end:
                        target_layers.append((name, module))

        # 复制权重
        for (source_name, source_module), (target_name, target_module) in zip(source_layers, target_layers):
            try:
                if isinstance(source_module, nn.Module) and isinstance(target_module, nn.Module):
                    # 检查是否为 Conv2d 层
                    if isinstance(source_module, nn.Conv2d) and isinstance(target_module, nn.Conv2d):
                        # 检查通道数
                        if target_module.in_channels == 1 and source_module.in_channels == 3:
                            # 如果目标层的输入通道数为1，源层输入通道数为3，则取平均
                            new_weight = source_module.weight.mean(dim=1, keepdim=True)
                            target_module.weight = nn.Parameter(new_weight)
                            print(f"复制并修改权重: {source_name} -> {target_name}")
                        elif target_module.out_channels > source_module.out_channels:
                            # 如果目标层的输出通道数大于源层，则复制源层的通道
                            new_weight = torch.cat([source_module.weight] * (target_module.out_channels // source_module.out_channels), dim=0)
                            target_module.weight = nn.Parameter(new_weight)
                            print(f"复制并增加通道数: {source_name} -> {target_name}")
                        else:
                            # 直接复制权重
                            target_module.load_state_dict(source_module.state_dict())
                            print(f"复制权重: {source_name} -> {target_name}")
                    else:
                        # 直接复制其他类型的模块
                        target_module.load_state_dict(source_module.state_dict())
                        print(f"复制权重: {source_name} -> {target_name}")
            except:
                print(f"复制权重失败: {source_name} -> {target_name}")
                continue


    # 定义要设置为0的层名称
    zero_layers = ['model.8','model.15','model.21','model.27']  # 示例：将目标模型的 model.38 层的权重设置为0

    for layer_name in zero_layers:
        # 查找目标层
        target_layer = None
        for name, module in target_model.named_modules():
            if name == layer_name:
                target_layer = module
                break

        # 确保找到目标层
        assert target_layer is not None, f"未找到目标层 {layer_name}"
        # print(type(target_layer))
        # 将目标层的权重设置为0
        if isinstance(target_layer, nn.Conv2d):
            nn.init.zeros_(target_layer.weight)
            if target_layer.bias is not None:
                nn.init.zeros_(target_layer.bias)
            print(f"将层 {layer_name} 的权重设置为0")
            print("2D Conv Weights (sum):", target_layer.weight.sum().item())

    # 创建元数据字典
    metadata = {
        'date': datetime.datetime.now().isoformat(),
        'version': '8.2.100',
        'license': 'AGPL-3.0 License (https://ultralytics.com/license)',
        'docs': 'https://docs.ultralytics.com',
        'epoch': 300,
        'best_fitness': None,
        'model': target_model
    }
    # print(target_model)
    # 保存模型和元数据
    torch.save(metadata, output_model_path)
    print(f"最终模型已成功保存到 {output_model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Step1/Step2 weights into MCF initialization checkpoint")
    parser.add_argument(
        "--source-model-path",
        type=str,
        default="runs/GAIIC2024/GAIIC2024-yolo11x-RGB-main/weights/best.pt",
        help="Step1 single-modal best.pt path",
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        default="runs/GAIIC2024/GAIIC2024-yolo11x-RGBT-MCF-template/weights/best.pt",
        help="Step2 MCF template best.pt path",
    )
    parser.add_argument(
        "--output-model-path",
        type=str,
        default="runs/GAIIC2024/GAIIC2024-yolo11x-RGBT-MCF.pt",
        help="Output converted MCF checkpoint path",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    copy_and_modify_layers(
        source_model_path=args.source_model_path,
        target_model_path=args.target_model_path,
        output_model_path=args.output_model_path,
    )
