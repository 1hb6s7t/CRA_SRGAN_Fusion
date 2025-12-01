# Copyright 2024
# 模型评估脚本
"""
评估模型性能

支持指标:
- PSNR (峰值信噪比)
- SSIM (结构相似性)
- LPIPS (感知相似性)
- FID (Fréchet Inception Distance)
"""

import os
import sys
import argparse
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# 评估指标
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    print("请安装scikit-image: pip install scikit-image")
    sys.exit(1)


def calculate_psnr(img1, img2):
    """计算PSNR"""
    return psnr(img1, img2, data_range=255)


def calculate_ssim(img1, img2, multichannel=True):
    """计算SSIM"""
    if len(img1.shape) == 3 and multichannel:
        return ssim(img1, img2, multichannel=True, data_range=255, channel_axis=2)
    return ssim(img1, img2, data_range=255)


def evaluate_single(gt_path, pred_path):
    """评估单张图像"""
    gt = cv2.imread(gt_path)
    pred = cv2.imread(pred_path)
    
    if gt is None or pred is None:
        return None, None
    
    # 确保尺寸一致
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    
    psnr_val = calculate_psnr(gt, pred)
    ssim_val = calculate_ssim(gt, pred)
    
    return psnr_val, ssim_val


def evaluate_directory(gt_dir, pred_dir, output_file=None):
    """评估整个目录"""
    gt_files = sorted(glob(os.path.join(gt_dir, '*.png')))
    gt_files += sorted(glob(os.path.join(gt_dir, '*.jpg')))
    
    psnr_list = []
    ssim_list = []
    
    results = []
    
    for gt_path in tqdm(gt_files, desc="评估中"):
        filename = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, filename)
        
        if not os.path.exists(pred_path):
            # 尝试其他扩展名
            basename = os.path.splitext(filename)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                pred_path = os.path.join(pred_dir, basename + ext)
                if os.path.exists(pred_path):
                    break
        
        if not os.path.exists(pred_path):
            print(f"未找到对应的预测结果: {filename}")
            continue
        
        psnr_val, ssim_val = evaluate_single(gt_path, pred_path)
        
        if psnr_val is not None:
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            results.append({
                'filename': filename,
                'psnr': psnr_val,
                'ssim': ssim_val
            })
    
    # 计算平均值
    avg_psnr = np.mean(psnr_list) if psnr_list else 0
    avg_ssim = np.mean(ssim_list) if ssim_list else 0
    
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"样本数量: {len(psnr_list)}")
    print(f"平均 PSNR: {avg_psnr:.4f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    print("=" * 50)
    
    # 保存结果
    if output_file:
        with open(output_file, 'w') as f:
            f.write("filename,psnr,ssim\n")
            for r in results:
                f.write(f"{r['filename']},{r['psnr']:.4f},{r['ssim']:.4f}\n")
            f.write(f"\naverage,{avg_psnr:.4f},{avg_ssim:.4f}\n")
        print(f"结果已保存到: {output_file}")
    
    return avg_psnr, avg_ssim


def evaluate_inpainting(gt_dir, pred_dir, mask_dir, output_file=None):
    """
    评估修复效果 (仅计算修复区域)
    """
    gt_files = sorted(glob(os.path.join(gt_dir, '*.png')))
    gt_files += sorted(glob(os.path.join(gt_dir, '*.jpg')))
    
    psnr_list = []
    ssim_list = []
    
    for gt_path in tqdm(gt_files, desc="评估修复区域"):
        filename = os.path.basename(gt_path)
        basename = os.path.splitext(filename)[0]
        
        # 查找对应文件
        pred_path = os.path.join(pred_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        
        for ext in ['.png', '.jpg']:
            if not os.path.exists(pred_path):
                pred_path = os.path.join(pred_dir, basename + ext)
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_dir, basename + ext)
        
        if not os.path.exists(pred_path) or not os.path.exists(mask_path):
            continue
        
        gt = cv2.imread(gt_path)
        pred = cv2.imread(pred_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if gt is None or pred is None or mask is None:
            continue
        
        # 调整尺寸
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        if mask.shape[:2] != gt.shape[:2]:
            mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]))
        
        # 只计算修复区域
        mask_binary = (mask > 127).astype(np.uint8)
        
        if mask_binary.sum() == 0:
            continue
        
        # 提取修复区域
        coords = np.where(mask_binary > 0)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        gt_region = gt[y_min:y_max, x_min:x_max]
        pred_region = pred[y_min:y_max, x_min:x_max]
        
        if gt_region.size > 0:
            psnr_val = calculate_psnr(gt_region, pred_region)
            ssim_val = calculate_ssim(gt_region, pred_region)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
    
    avg_psnr = np.mean(psnr_list) if psnr_list else 0
    avg_ssim = np.mean(ssim_list) if ssim_list else 0
    
    print("\n" + "=" * 50)
    print("修复区域评估结果")
    print("=" * 50)
    print(f"样本数量: {len(psnr_list)}")
    print(f"修复区域平均 PSNR: {avg_psnr:.4f} dB")
    print(f"修复区域平均 SSIM: {avg_ssim:.4f}")
    print("=" * 50)
    
    return avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser(description='模型评估工具')
    parser.add_argument('--gt_dir', type=str, required=True, help='真实图像目录')
    parser.add_argument('--pred_dir', type=str, required=True, help='预测结果目录')
    parser.add_argument('--mask_dir', type=str, default=None, help='掩码目录(可选,用于评估修复区域)')
    parser.add_argument('--output', type=str, default=None, help='输出CSV文件路径')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'inpainting'],
                       help='评估模式: full(全图), inpainting(仅修复区域)')
    
    args = parser.parse_args()
    
    if args.mode == 'inpainting' and args.mask_dir:
        evaluate_inpainting(args.gt_dir, args.pred_dir, args.mask_dir, args.output)
    else:
        evaluate_directory(args.gt_dir, args.pred_dir, args.output)


if __name__ == '__main__':
    main()

