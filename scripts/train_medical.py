# Copyright 2024
# 医学影像修复专用训练脚本
"""
医学影像修复模型训练

特殊优化:
1. 高像素精度 (增加L1损失权重)
2. 保守的对抗训练
3. 支持灰度图像
4. 适合X光、CT、MRI等医学影像修复
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import random

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import context, Tensor, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean, _get_parallel_mode
from mindspore.context import ParallelMode
import mindspore.ops.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import CRASRGANConfig
from src.models.fusion_generator import CRASRGANGenerator
from src.models.fusion_discriminator import MultiScaleDiscriminator
from src.loss.hybrid_loss import GeneratorLoss, DiscriminatorLoss


class MedicalDataset:
    """医学影像数据集"""
    
    def __init__(self, image_dir, mask_dir, img_size=512, augment=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        
        from glob import glob
        self.image_files = sorted(glob(os.path.join(image_dir, '*.png')))
        self.image_files += sorted(glob(os.path.join(image_dir, '*.jpg')))
        self.image_files += sorted(glob(os.path.join(image_dir, '*.dcm')))
        
        self.mask_files = sorted(glob(os.path.join(mask_dir, '*.png')))
        self.mask_files += sorted(glob(os.path.join(mask_dir, '*.jpg')))
        
        print(f"[数据集] 加载了 {len(self.image_files)} 张医学影像")
        print(f"[数据集] 加载了 {len(self.mask_files)} 个掩码")
    
    def __len__(self):
        return max(1, len(self.image_files))
    
    def __getitem__(self, idx):
        if len(self.image_files) == 0:
            img = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
            mask = np.zeros((1, self.img_size, self.img_size), dtype=np.float32)
            return img, mask
        
        idx = idx % len(self.image_files)
        
        # 读取图像
        img_path = self.image_files[idx]
        
        if img_path.lower().endswith('.dcm'):
            try:
                import pydicom
                dcm = pydicom.dcmread(img_path)
                img = dcm.pixel_array.astype(np.float32)
                img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                print(f"读取DICOM失败: {e}")
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.imread(img_path)
            if img is not None:
                # 处理灰度图
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # 读取掩码
        if len(self.mask_files) > 0:
            mask_idx = random.randint(0, len(self.mask_files) - 1)
            mask = cv2.imread(self.mask_files[mask_idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                mask = cv2.resize(mask, (self.img_size, self.img_size))
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # 医学影像只做轻微增强
        if self.augment:
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
        
        # 归一化
        img = img.astype(np.float32) / 127.5 - 1
        mask = mask.astype(np.float32) / 255.0
        
        img = img.transpose(2, 0, 1)
        mask = mask[np.newaxis, ...]
        
        return img, mask


class TrainOneStepG(nn.Cell):
    """生成器单步训练"""
    
    def __init__(self, g_loss, optimizer, sens=1.0):
        super(TrainOneStepG, self).__init__(auto_prefix=True)
        self.optimizer = optimizer
        self.g_loss = g_loss
        self.g_loss.generator.set_grad()
        self.g_loss.generator.set_train()
        self.g_loss.discriminator.set_grad(False)
        self.g_loss.discriminator.set_train(False)
        
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.fill = ops.Fill()
        self.dtype = ops.DType()
        self.shape = ops.Shape()
        self.grad_reducer = F.identity
        
        self.reducer_flag = False
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)
    
    def construct(self, real, x, mask):
        weights = self.weights
        loss = self.g_loss(real, x, mask)
        sens = self.fill(self.dtype(loss), self.shape(loss), self.sens)
        grads = self.grad(self.g_loss, weights)(real, x, mask, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss


class TrainOneStepD(nn.Cell):
    """判别器单步训练"""
    
    def __init__(self, d_loss, optimizer, sens=1.0):
        super(TrainOneStepD, self).__init__(auto_prefix=True)
        self.optimizer = optimizer
        self.d_loss = d_loss
        self.d_loss.discriminator.set_grad()
        self.d_loss.discriminator.set_train()
        self.d_loss.generator.set_grad(False)
        self.d_loss.generator.set_train(False)
        
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.fill = ops.Fill()
        self.dtype = ops.DType()
        self.shape = ops.Shape()
        self.grad_reducer = F.identity
        
        self.reducer_flag = False
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)
    
    def construct(self, real, x, mask):
        weights = self.weights
        loss = self.d_loss(real, x, mask)
        sens = self.fill(self.dtype(loss), self.shape(loss), self.sens)
        grads = self.grad(self.d_loss, weights)(real, x, mask, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss


def train():
    """医学影像训练主函数"""
    
    parser = argparse.ArgumentParser(description='医学影像修复模型训练')
    parser.add_argument('--image_dir', type=str, required=True, help='图像目录')
    parser.add_argument('--mask_dir', type=str, required=True, help='掩码目录')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', type=str, default='GPU', choices=['GPU', 'Ascend', 'CPU'])
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/medical')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=20)
    args = parser.parse_args()
    
    # 设置环境
    set_seed(2024)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device)
    if args.device in ['GPU', 'Ascend']:
        context.set_context(device_id=args.device_id)
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'logs'), exist_ok=True)
    
    print("=" * 70)
    print("                    医学影像修复模型训练")
    print("=" * 70)
    print(f"设备: {args.device}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"总轮数: {args.epochs}")
    print("=" * 70)
    
    # 配置 - 医学影像特殊配置
    config = CRASRGANConfig()
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    
    # 医学影像: 高精度重建,最小化GAN影响
    config.training.l1_weight = 2.0
    config.training.perceptual_weight = 0.05
    config.training.style_weight = 0.01
    config.training.edge_weight = 0.2
    config.training.adversarial_weight = 0.0001
    
    print("\n[配置] 医学影像损失权重:")
    print(f"  L1 (像素精度): {config.training.l1_weight}")
    print(f"  感知: {config.training.perceptual_weight}")
    print(f"  风格: {config.training.style_weight}")
    print(f"  边缘: {config.training.edge_weight}")
    print(f"  对抗: {config.training.adversarial_weight}")
    
    # 数据集
    print("\n[数据] 加载数据集...")
    train_dataset = MedicalDataset(args.image_dir, args.mask_dir, 512, True)
    train_loader = ds.GeneratorDataset(
        train_dataset,
        column_names=['image', 'mask'],
        shuffle=True,
        num_parallel_workers=4
    ).batch(args.batch_size, drop_remainder=True)
    
    dataset_size = train_loader.get_dataset_size()
    print(f"[数据] 每轮迭代数: {dataset_size}")
    
    # 模型
    print("\n[模型] 创建模型...")
    generator = CRASRGANGenerator(config)
    discriminator = MultiScaleDiscriminator(in_channels=3, num_scales=3)
    
    if args.resume:
        print(f"[模型] 加载权重: {args.resume}")
        param_dict = load_checkpoint(args.resume)
        load_param_into_net(generator, param_dict)
    
    # 优化器
    total_steps = args.epochs * dataset_size
    lr_schedule = nn.exponential_decay_lr(args.lr, 0.5, total_steps, dataset_size, 50)
    
    optimizer_g = nn.Adam(generator.trainable_params(), lr_schedule, 0.9, 0.999)
    optimizer_d = nn.Adam(discriminator.trainable_params(), lr_schedule, 0.9, 0.999)
    
    # 损失
    g_loss_fn = GeneratorLoss(generator, discriminator, config)
    d_loss_fn = DiscriminatorLoss(generator, discriminator, config)
    
    train_g = TrainOneStepG(g_loss_fn, optimizer_g)
    train_d = TrainOneStepD(d_loss_fn, optimizer_d)
    
    # 训练
    print("\n[训练] 开始训练...")
    train_g.set_train()
    train_d.set_train()
    
    best_loss = float('inf')
    log_file = open(os.path.join(args.save_dir, 'logs', 'train_log.txt'), 'w')
    
    for epoch in range(args.epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        batch_count = 0
        start_time = time.time()
        
        for batch in train_loader.create_dict_iterator():
            real = batch['image']
            mask = batch['mask']
            x = real * (1 - mask)
            
            d_loss = train_d(real, x, mask)
            g_loss = train_g(real, x, mask)
            
            epoch_g_loss += float(g_loss.asnumpy())
            epoch_d_loss += float(d_loss.asnumpy())
            batch_count += 1
            
            if batch_count % args.log_interval == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_count}/{dataset_size}, "
                      f"G_loss: {float(g_loss.asnumpy()):.4f}, "
                      f"D_loss: {float(d_loss.asnumpy()):.4f}")
        
        avg_g_loss = epoch_g_loss / max(batch_count, 1)
        avg_d_loss = epoch_d_loss / max(batch_count, 1)
        elapsed = time.time() - start_time
        
        log_msg = f"Epoch [{epoch+1}/{args.epochs}], G_loss: {avg_g_loss:.6f}, D_loss: {avg_d_loss:.6f}, Time: {elapsed:.2f}s"
        print(log_msg)
        log_file.write(log_msg + '\n')
        log_file.flush()
        
        if avg_g_loss < best_loss:
            best_loss = avg_g_loss
            save_checkpoint(generator, os.path.join(args.save_dir, 'best_generator.ckpt'))
            print(f"  [保存] 最佳模型已保存")
        
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(generator, os.path.join(args.save_dir, f'generator_epoch{epoch+1}.ckpt'))
    
    log_file.close()
    print("\n[完成] 医学影像修复模型训练完成!")


if __name__ == '__main__':
    train()

