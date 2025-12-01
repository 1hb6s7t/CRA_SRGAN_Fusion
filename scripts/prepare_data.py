# Copyright 2024
# 数据集准备脚本
"""
数据集准备工具

支持:
1. 文物修复数据准备
2. 医学影像数据准备
3. 自动生成破损掩码
4. 数据增强
"""

import os
import sys
import argparse
import cv2
import numpy as np
from glob import glob
import random
from tqdm import tqdm


def create_damage_mask(image_shape, damage_type='mixed', intensity='medium'):
    """
    创建模拟破损掩码
    
    Args:
        image_shape: (H, W) 图像尺寸
        damage_type: 破损类型
            - 'scratch': 划痕
            - 'missing': 缺失区域
            - 'stain': 污渍
            - 'crack': 裂纹
            - 'artifact': 设备伪影 (医学影像)
            - 'mixed': 混合类型
        intensity: 破损强度 'light', 'medium', 'heavy'
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 根据强度调整参数
    intensity_params = {
        'light': {'num_elements': (1, 3), 'size_factor': 0.5},
        'medium': {'num_elements': (2, 5), 'size_factor': 1.0},
        'heavy': {'num_elements': (4, 8), 'size_factor': 1.5}
    }
    params = intensity_params.get(intensity, intensity_params['medium'])
    
    if damage_type == 'scratch' or damage_type == 'mixed':
        # 模拟划痕
        num_scratches = random.randint(*params['num_elements'])
        for _ in range(num_scratches):
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (random.randint(0, w), random.randint(0, h))
            thickness = int(random.randint(2, 8) * params['size_factor'])
            cv2.line(mask, pt1, pt2, 255, thickness)
    
    if damage_type == 'missing' or damage_type == 'mixed':
        # 模拟缺失区域 (不规则多边形)
        num_missing = random.randint(1, max(1, params['num_elements'][0]))
        for _ in range(num_missing):
            num_points = random.randint(4, 8)
            center_x = random.randint(w//4, 3*w//4)
            center_y = random.randint(h//4, 3*h//4)
            points = []
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                radius = int(random.randint(20, min(h, w) // 6) * params['size_factor'])
                x = int(center_x + radius * np.cos(angle) + random.randint(-10, 10))
                y = int(center_y + radius * np.sin(angle) + random.randint(-10, 10))
                points.append([x, y])
            points = np.array([points], dtype=np.int32)
            cv2.fillPoly(mask, points, 255)
    
    if damage_type == 'stain' or damage_type == 'mixed':
        # 模拟污渍 (圆形斑点)
        num_stains = random.randint(*params['num_elements'])
        for _ in range(num_stains):
            center = (random.randint(0, w), random.randint(0, h))
            radius = int(random.randint(10, 50) * params['size_factor'])
            cv2.circle(mask, center, radius, 255, -1)
    
    if damage_type == 'crack' or damage_type == 'mixed':
        # 模拟裂纹 (随机游走)
        num_cracks = random.randint(1, max(1, params['num_elements'][0]))
        for _ in range(num_cracks):
            x, y = random.randint(0, w), random.randint(0, h)
            steps = int(random.randint(50, 200) * params['size_factor'])
            for _ in range(steps):
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                x = max(0, min(w-1, x + dx))
                y = max(0, min(h-1, y + dy))
                cv2.circle(mask, (x, y), random.randint(1, 3), 255, -1)
    
    if damage_type == 'artifact':
        # 医学影像伪影 (条纹状)
        num_artifacts = random.randint(*params['num_elements'])
        for _ in range(num_artifacts):
            if random.random() > 0.5:
                y = random.randint(0, h-1)
                thickness = int(random.randint(5, 20) * params['size_factor'])
                cv2.rectangle(mask, (0, y), (w, y + thickness), 255, -1)
            else:
                x = random.randint(0, w-1)
                thickness = int(random.randint(5, 20) * params['size_factor'])
                cv2.rectangle(mask, (x, 0), (x + thickness, h), 255, -1)
    
    # 模糊处理,使边缘更自然
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask


def prepare_cultural_relics_dataset(input_dir, output_dir, target_size=512, 
                                     masks_per_image=3, split_ratio=0.9):
    """
    准备文物修复数据集
    
    Args:
        input_dir: 原始图像目录
        output_dir: 输出目录
        target_size: 目标尺寸
        masks_per_image: 每张图像生成的掩码数量
        split_ratio: 训练集比例
    """
    print("=" * 60)
    print("准备文物修复数据集")
    print("=" * 60)
    
    # 创建目录
    train_img_dir = os.path.join(output_dir, 'train', 'images')
    train_mask_dir = os.path.join(output_dir, 'train', 'masks')
    val_img_dir = os.path.join(output_dir, 'val', 'images')
    val_mask_dir = os.path.join(output_dir, 'val', 'masks')
    
    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 获取所有图像
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
        image_files.extend(glob(os.path.join(input_dir, ext)))
        image_files.extend(glob(os.path.join(input_dir, ext.upper())))
        image_files.extend(glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    image_files = list(set(image_files))  # 去重
    random.shuffle(image_files)
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 分割训练/验证集
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"训练集: {len(train_files)} 张")
    print(f"验证集: {len(val_files)} 张")
    
    # 处理训练集
    print("\n处理训练集...")
    process_images(train_files, train_img_dir, train_mask_dir, 
                   target_size, masks_per_image, 'mixed')
    
    # 处理验证集
    print("\n处理验证集...")
    process_images(val_files, val_img_dir, val_mask_dir, 
                   target_size, masks_per_image, 'mixed')
    
    print("\n文物修复数据集准备完成!")
    print(f"数据保存在: {output_dir}")


def prepare_medical_dataset(input_dir, output_dir, target_size=512, 
                            masks_per_image=2, split_ratio=0.9, is_grayscale=True):
    """
    准备医学影像数据集
    """
    print("=" * 60)
    print("准备医学影像数据集")
    print("=" * 60)
    
    # 创建目录
    train_img_dir = os.path.join(output_dir, 'train', 'images')
    train_mask_dir = os.path.join(output_dir, 'train', 'masks')
    val_img_dir = os.path.join(output_dir, 'val', 'images')
    val_mask_dir = os.path.join(output_dir, 'val', 'masks')
    
    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 获取所有图像
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.dcm']:
        image_files.extend(glob(os.path.join(input_dir, ext)))
        image_files.extend(glob(os.path.join(input_dir, ext.upper())))
        image_files.extend(glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    image_files = list(set(image_files))
    random.shuffle(image_files)
    
    print(f"找到 {len(image_files)} 张医学影像")
    
    # 分割
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"训练集: {len(train_files)} 张")
    print(f"验证集: {len(val_files)} 张")
    
    # 处理
    print("\n处理训练集...")
    process_images(train_files, train_img_dir, train_mask_dir, 
                   target_size, masks_per_image, 'artifact', is_grayscale)
    
    print("\n处理验证集...")
    process_images(val_files, val_img_dir, val_mask_dir, 
                   target_size, masks_per_image, 'artifact', is_grayscale)
    
    print("\n医学影像数据集准备完成!")


def process_images(image_files, img_out_dir, mask_out_dir, 
                   target_size, masks_per_image, damage_type, is_grayscale=False):
    """处理图像并生成掩码"""
    
    for idx, img_path in enumerate(tqdm(image_files, desc="处理图像")):
        try:
            # 读取图像
            if img_path.lower().endswith('.dcm'):
                try:
                    import pydicom
                    dcm = pydicom.dcmread(img_path)
                    img = dcm.pixel_array.astype(np.float32)
                    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                except ImportError:
                    print("需要安装pydicom: pip install pydicom")
                    continue
            else:
                if is_grayscale:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # 调整尺寸
            img = cv2.resize(img, (target_size, target_size))
            
            # 生成多个掩码
            intensities = ['light', 'medium', 'heavy']
            for mask_idx in range(masks_per_image):
                intensity = random.choice(intensities)
                mask = create_damage_mask(img.shape, damage_type, intensity)
                
                # 保存
                img_name = f"{idx:06d}_{mask_idx}.png"
                cv2.imwrite(os.path.join(img_out_dir, img_name), img)
                cv2.imwrite(os.path.join(mask_out_dir, img_name), mask)
                
        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='数据集准备工具')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['cultural_relics', 'medical'],
                       help='数据集类型')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='原始图像目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--target_size', type=int, default=512,
                       help='目标图像尺寸')
    parser.add_argument('--masks_per_image', type=int, default=3,
                       help='每张图像生成的掩码数量')
    parser.add_argument('--split_ratio', type=float, default=0.9,
                       help='训练集比例')
    parser.add_argument('--grayscale', action='store_true',
                       help='是否为灰度图像')
    
    args = parser.parse_args()
    
    if args.mode == 'cultural_relics':
        prepare_cultural_relics_dataset(
            args.input_dir,
            args.output_dir,
            args.target_size,
            args.masks_per_image,
            args.split_ratio
        )
    elif args.mode == 'medical':
        prepare_medical_dataset(
            args.input_dir,
            args.output_dir,
            args.target_size,
            args.masks_per_image,
            args.split_ratio,
            args.grayscale
        )


if __name__ == '__main__':
    main()

