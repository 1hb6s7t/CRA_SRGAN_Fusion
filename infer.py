# Copyright 2024
# CRA-SRGAN Fusion Model Inference Script
"""
图像修复与超高清化一体化模型推理脚本

支持功能:
- 单张图像推理
- 批量推理
- 8K超高分辨率推理
- 分块推理优化
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

import mindspore
from mindspore import context, Tensor, load_checkpoint, load_param_into_net

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config.config import CRASRGANConfig, get_default_config
from src.models.fusion_generator import CRASRGANGenerator
from src.utils.inference_8k import InferenceEngine, TileInference


def load_image(image_path, target_size=None):
    """
    加载并预处理图像
    
    Args:
        image_path: 图像路径
        target_size: 目标尺寸 (H, W), None表示保持原尺寸
        
    Returns:
        image_tensor: 归一化的图像张量
        original_size: 原始尺寸
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # 调整尺寸
    if target_size is not None:
        image = cv2.resize(image, (target_size[1], target_size[0]))
    
    # 归一化到 [-1, 1]
    image = image.astype(np.float32) / 127.5 - 1
    
    # 转换为 (1, C, H, W)
    image = image.transpose(2, 0, 1)[np.newaxis, ...]
    
    return Tensor(image, mindspore.float32), original_size


def load_mask(mask_path, target_size):
    """
    加载并预处理掩码
    
    Args:
        mask_path: 掩码路径
        target_size: 目标尺寸 (H, W)
        
    Returns:
        mask_tensor: 掩码张量 (1=破损, 0=有效)
    """
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (target_size[1], target_size[0]))
    else:
        # 如果没有提供掩码,创建全零掩码 (无破损)
        mask = np.zeros((target_size[0], target_size[1]), dtype=np.float32)
    
    # 归一化到 [0, 1], 1表示破损
    mask = mask.astype(np.float32) / 255.0
    
    # 转换为 (1, 1, H, W)
    mask = mask[np.newaxis, np.newaxis, ...]
    
    return Tensor(mask, mindspore.float32)


def save_image(tensor, output_path, original_size=None):
    """
    保存输出图像
    
    Args:
        tensor: 输出张量
        output_path: 保存路径
        original_size: 原始尺寸 (可选,用于恢复原尺寸)
    """
    # 转换为numpy
    image = tensor.asnumpy()[0]  # (C, H, W)
    image = image.transpose(1, 2, 0)  # (H, W, C)
    
    # 反归一化
    image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    # 恢复原尺寸 (如果需要)
    if original_size is not None:
        image = cv2.resize(image, (original_size[1], original_size[0]))
    
    # BGR转换
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 保存
    cv2.imwrite(output_path, image)
    print(f"Saved to: {output_path}")


def infer_single(model, image_path, mask_path, output_path, config, 
                 use_8k_mode=False):
    """
    单张图像推理
    
    Args:
        model: 推理模型
        image_path: 输入图像路径
        mask_path: 掩码路径
        output_path: 输出路径
        config: 配置
        use_8k_mode: 是否使用8K推理模式
    """
    print(f"\nProcessing: {image_path}")
    
    # 加载图像
    if use_8k_mode:
        image, original_size = load_image(image_path, target_size=None)
    else:
        image, original_size = load_image(image_path, target_size=(512, 512))
    
    # 加载掩码
    mask = load_mask(mask_path, (image.shape[2], image.shape[3]))
    
    # 创建被破损的输入
    x = image * (1 - mask)
    
    print(f"  Input size: {image.shape[2]}x{image.shape[3]}")
    
    # 推理
    start_time = time.time()
    
    if use_8k_mode:
        # 使用8K推理引擎
        engine = InferenceEngine(model, config)
        output = engine.infer(x, mask, mode='auto')
    else:
        # 直接推理
        coarse_out, refine_out, sr_out, final_out, _ = model(x, mask)
        output = final_out
    
    elapsed = time.time() - start_time
    print(f"  Inference time: {elapsed:.3f}s")
    print(f"  Output size: {output.shape[2]}x{output.shape[3]}")
    
    # 保存结果
    save_image(output, output_path)
    
    # 可选: 保存中间结果
    output_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    
    if not use_8k_mode:
        # 保存粗修复结果
        save_image(coarse_out, os.path.join(output_dir, f"{base_name}_coarse.png"))
        # 保存细修复结果
        save_image(refine_out, os.path.join(output_dir, f"{base_name}_refine.png"))


def infer_batch(model, input_dir, mask_dir, output_dir, config, use_8k_mode=False):
    """
    批量推理
    
    Args:
        model: 推理模型
        input_dir: 输入图像目录
        mask_dir: 掩码目录
        output_dir: 输出目录
        config: 配置
        use_8k_mode: 是否使用8K推理模式
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        import glob
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f"Found {len(image_files)} images")
    
    total_time = 0
    for image_path in image_files:
        # 构建掩码路径
        base_name = os.path.basename(image_path)
        mask_path = os.path.join(mask_dir, base_name) if mask_dir else None
        
        # 构建输出路径
        output_path = os.path.join(output_dir, f"output_{base_name}")
        
        # 推理
        start = time.time()
        infer_single(model, image_path, mask_path, output_path, config, use_8k_mode)
        total_time += time.time() - start
    
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/len(image_files):.3f}s")


def demo_8k_inference(model, config):
    """
    8K推理演示
    """
    print("\n" + "=" * 60)
    print("8K Inference Demo")
    print("=" * 60)
    
    # 创建测试数据
    print("Creating test data (512x512 input -> 8192x8192 output)...")
    test_image = Tensor(np.random.randn(1, 3, 512, 512), mindspore.float32)
    test_mask = Tensor(np.random.randn(1, 1, 512, 512) > 0.5, mindspore.float32)
    
    # 创建推理引擎
    engine = InferenceEngine(model, config)
    
    # 执行推理
    print("Running inference...")
    start = time.time()
    output = engine.infer(test_image, test_mask, mode='tile')
    elapsed = time.time() - start
    
    print(f"Input size: 512x512")
    print(f"Output size: {output.shape[2]}x{output.shape[3]}")
    print(f"Inference time: {elapsed:.2f}s")
    print(f"Upscale factor: {config.model.srgan_upscale_factor}x")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CRA-SRGAN Inference')
    
    # 输入输出参数
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--mask', '-m', type=str, default=None,
                       help='Mask image path or directory')
    parser.add_argument('--output', '-o', type=str, default='./output',
                       help='Output path or directory')
    
    # 模型参数
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Model checkpoint path')
    parser.add_argument('--device', type=str, default='GPU',
                       choices=['GPU', 'Ascend', 'CPU'])
    parser.add_argument('--device_id', type=int, default=0)
    
    # 推理模式
    parser.add_argument('--mode', type=str, default='normal',
                       choices=['normal', '8k', 'demo'],
                       help='Inference mode')
    parser.add_argument('--tile_size', type=int, default=512,
                       help='Tile size for 8K inference')
    
    args = parser.parse_args()
    
    # 设置环境
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device)
    if args.device in ['GPU', 'Ascend']:
        context.set_context(device_id=args.device_id)
    
    # 加载配置
    config = get_default_config()
    config.inference.tile_size = args.tile_size
    
    # 创建模型
    print("Loading model...")
    model = CRASRGANGenerator(config)
    
    # 加载权重
    param_dict = load_checkpoint(args.checkpoint)
    load_param_into_net(model, param_dict)
    model.set_train(False)
    print(f"Loaded checkpoint: {args.checkpoint}")
    
    # 执行推理
    if args.mode == 'demo':
        demo_8k_inference(model, config)
    elif os.path.isdir(args.input):
        # 批量推理
        infer_batch(model, args.input, args.mask, args.output, config,
                   use_8k_mode=(args.mode == '8k'))
    else:
        # 单张推理
        output_path = args.output
        if os.path.isdir(output_path):
            base_name = os.path.basename(args.input)
            output_path = os.path.join(output_path, f"output_{base_name}")
        
        infer_single(model, args.input, args.mask, output_path, config,
                    use_8k_mode=(args.mode == '8k'))
    
    print("\nDone!")


if __name__ == '__main__':
    main()

