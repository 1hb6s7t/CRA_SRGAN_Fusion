# Copyright 2024
# 可视化脚本
"""
修复结果可视化

功能:
1. 对比展示 (原图/破损/修复)
2. 差异图可视化
3. 批量生成对比图
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def visualize_single(original_path, mask_path, restored_path, save_path=None, show=True):
    """
    可视化单张图像的修复结果
    
    Args:
        original_path: 原始图像路径
        mask_path: 掩码路径
        restored_path: 修复结果路径
        save_path: 保存路径
        show: 是否显示
    """
    # 读取图像
    original = cv2.imread(original_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    restored = cv2.imread(restored_path)
    restored = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    
    # 调整尺寸一致
    h, w = original.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))
    if restored.shape[:2] != (h, w):
        restored = cv2.resize(restored, (w, h))
    
    # 创建破损图像
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask_binary = (mask / 255.0)[:, :, np.newaxis]
    damaged = (original * (1 - mask_binary) + mask_3ch * mask_binary).astype(np.uint8)
    
    # 计算差异图
    diff = np.abs(original.astype(np.float32) - restored.astype(np.float32))
    diff = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('原始图像', fontsize=14, fontproperties='SimHei')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('破损区域 (白色)', fontsize=14, fontproperties='SimHei')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(damaged)
    axes[0, 2].set_title('破损图像', fontsize=14, fontproperties='SimHei')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(restored)
    axes[1, 0].set_title('修复结果', fontsize=14, fontproperties='SimHei')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff)
    axes[1, 1].set_title('差异图 (与原图)', fontsize=14, fontproperties='SimHei')
    axes[1, 1].axis('off')
    
    # 局部放大 (修复区域)
    coords = np.where(mask > 127)
    if len(coords[0]) > 0:
        y_min, y_max = max(0, coords[0].min() - 20), min(h, coords[0].max() + 20)
        x_min, x_max = max(0, coords[1].min() - 20), min(w, coords[1].max() + 20)
        zoom_restored = restored[y_min:y_max, x_min:x_max]
        axes[1, 2].imshow(zoom_restored)
        axes[1, 2].set_title('修复区域放大', fontsize=14, fontproperties='SimHei')
    else:
        axes[1, 2].imshow(restored)
        axes[1, 2].set_title('修复结果', fontsize=14, fontproperties='SimHei')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_comparison(images_dict, save_path=None, show=True):
    """
    多方法对比可视化
    
    Args:
        images_dict: {'方法名': 图像路径, ...}
        save_path: 保存路径
        show: 是否显示
    """
    n = len(images_dict)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    
    for idx, (name, path) in enumerate(images_dict.items()):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if n == 1:
            ax = axes
        else:
            ax = axes[idx]
        
        ax.imshow(img)
        ax.set_title(name, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def batch_visualize(original_dir, mask_dir, restored_dir, output_dir, num_samples=10):
    """
    批量生成对比图
    
    Args:
        original_dir: 原始图像目录
        mask_dir: 掩码目录
        restored_dir: 修复结果目录
        output_dir: 输出目录
        num_samples: 采样数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    original_files = sorted(glob(os.path.join(original_dir, '*.png')))
    original_files += sorted(glob(os.path.join(original_dir, '*.jpg')))
    
    # 随机采样
    import random
    if len(original_files) > num_samples:
        original_files = random.sample(original_files, num_samples)
    
    for idx, orig_path in enumerate(original_files):
        filename = os.path.basename(orig_path)
        basename = os.path.splitext(filename)[0]
        
        mask_path = os.path.join(mask_dir, filename)
        restored_path = os.path.join(restored_dir, filename)
        
        # 尝试其他扩展名
        for ext in ['.png', '.jpg', '.jpeg']:
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_dir, basename + ext)
            if not os.path.exists(restored_path):
                restored_path = os.path.join(restored_dir, basename + ext)
        
        if not os.path.exists(mask_path) or not os.path.exists(restored_path):
            print(f"跳过: {filename} (找不到对应文件)")
            continue
        
        save_path = os.path.join(output_dir, f'comparison_{idx:03d}.png')
        visualize_single(orig_path, mask_path, restored_path, save_path, show=False)
    
    print(f"\n批量可视化完成! 结果保存在: {output_dir}")


def create_video(image_dir, output_path, fps=2):
    """
    将对比图序列创建为视频
    """
    images = sorted(glob(os.path.join(image_dir, '*.png')))
    
    if not images:
        print("未找到图像")
        return
    
    # 读取第一张图获取尺寸
    sample = cv2.imread(images[0])
    h, w = sample.shape[:2]
    
    # 创建视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img_path in images:
        img = cv2.imread(img_path)
        out.write(img)
    
    out.release()
    print(f"视频已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='结果可视化工具')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'batch', 'video'],
                       help='可视化模式')
    parser.add_argument('--original', type=str, help='原始图像/目录')
    parser.add_argument('--mask', type=str, help='掩码/目录')
    parser.add_argument('--restored', type=str, help='修复结果/目录')
    parser.add_argument('--output', type=str, help='输出路径/目录')
    parser.add_argument('--num_samples', type=int, default=10, help='批量模式采样数量')
    parser.add_argument('--fps', type=int, default=2, help='视频帧率')
    parser.add_argument('--no_show', action='store_true', help='不显示图像')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not all([args.original, args.mask, args.restored]):
            print("单图模式需要指定 --original, --mask, --restored")
            return
        visualize_single(
            args.original, args.mask, args.restored,
            args.output, show=not args.no_show
        )
    
    elif args.mode == 'batch':
        if not all([args.original, args.mask, args.restored, args.output]):
            print("批量模式需要指定 --original, --mask, --restored, --output")
            return
        batch_visualize(
            args.original, args.mask, args.restored,
            args.output, args.num_samples
        )
    
    elif args.mode == 'video':
        if not args.original or not args.output:
            print("视频模式需要指定 --original (图像目录) 和 --output")
            return
        create_video(args.original, args.output, args.fps)


if __name__ == '__main__':
    main()

