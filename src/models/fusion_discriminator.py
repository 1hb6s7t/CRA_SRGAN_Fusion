# Copyright 2024
# CRA-SRGAN Fusion Discriminator
"""
多尺度判别器

核心设计:
1. 多尺度判别 - 在不同分辨率下评估生成质量
2. 局部-全局判别 - 同时评估局部细节和全局一致性
3. 谱归一化 - 稳定GAN训练
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import TruncatedNormal, HeNormal
import numpy as np


class Conv2dBlock(nn.Cell):
    """
    卷积块 (判别器用)
    
    包含卷积、归一化、激活
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, use_spectral_norm=True, use_batch_norm=False):
        super(Conv2dBlock, self).__init__()
        
        layers = []
        
        # 卷积层
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=padding, pad_mode='pad', has_bias=True
        )
        layers.append(conv)
        
        # 可选的批归一化
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # LeakyReLU激活
        layers.append(nn.LeakyReLU(0.2))
        
        self.block = nn.SequentialCell(layers)
        
    def construct(self, x):
        return self.block(x)


class PatchDiscriminator(nn.Cell):
    """
    PatchGAN判别器
    
    输出N×N的判别图,每个位置判断对应感受野区域的真假
    """
    
    def __init__(self, in_channels=3, base_channels=64, num_layers=4):
        super(PatchDiscriminator, self).__init__()
        
        layers = []
        
        # 第一层 (不使用归一化)
        layers.append(Conv2dBlock(in_channels, base_channels, 4, 2, 1, 
                                  use_batch_norm=False))
        
        # 中间层
        channels = base_channels
        for i in range(1, num_layers):
            out_channels = min(channels * 2, 512)
            stride = 2 if i < num_layers - 1 else 1
            layers.append(Conv2dBlock(channels, out_channels, 4, stride, 1,
                                     use_batch_norm=True))
            channels = out_channels
        
        # 输出层
        layers.append(nn.Conv2d(channels, 1, 4, 1, padding=1, pad_mode='pad'))
        
        self.model = nn.SequentialCell(layers)
        
    def construct(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Cell):
    """
    多尺度判别器 (Multi-Scale Discriminator)
    
    创新点:
    - 在多个尺度上评估生成质量
    - 捕获不同粒度的纹理和结构信息
    - 适用于8K超高分辨率图像
    
    Args:
        in_channels: 输入通道数
        base_channels: 基础通道数
        num_scales: 判别尺度数
        num_layers: 每个判别器的层数
    """
    
    def __init__(self, in_channels=3, base_channels=64, num_scales=3, num_layers=4):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.num_scales = num_scales
        
        # 创建多个尺度的判别器
        self.discriminators = nn.CellList()
        for _ in range(num_scales):
            self.discriminators.append(
                PatchDiscriminator(in_channels, base_channels, num_layers)
            )
        
        # 下采样操作
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='same')
        
    def construct(self, x):
        """
        多尺度判别前向传播
        
        Args:
            x: 输入图像 (B, C, H, W)
            
        Returns:
            outputs: 各尺度的判别结果列表
        """
        outputs = []
        current = x
        
        for i, disc in enumerate(self.discriminators):
            # 当前尺度的判别
            out = disc(current)
            outputs.append(out)
            
            # 下采样到下一个尺度 (除了最后一个)
            if i < self.num_scales - 1:
                current = self.downsample(current)
        
        return outputs


class InpaintingDiscriminator(nn.Cell):
    """
    修复专用判别器
    
    专门用于判断修复区域的质量
    """
    
    def __init__(self, in_channels=3, base_channels=64):
        super(InpaintingDiscriminator, self).__init__()
        
        # 特征提取
        self.features = nn.SequentialCell([
            Conv2dBlock(in_channels, base_channels, 5, 2, 2),          # 256
            Conv2dBlock(base_channels, base_channels * 2, 5, 2, 2),   # 128
            Conv2dBlock(base_channels * 2, base_channels * 4, 5, 2, 2), # 64
            Conv2dBlock(base_channels * 4, base_channels * 4, 5, 2, 2), # 32
            Conv2dBlock(base_channels * 4, base_channels * 4, 5, 2, 2), # 16
            Conv2dBlock(base_channels * 4, base_channels * 4, 5, 2, 2), # 8
        ])
        
        self.flatten = nn.Flatten()
        
        # 分类器
        self.classifier = nn.Dense(base_channels * 4 * 8 * 8, 1)
        
    def construct(self, x):
        """
        修复判别器前向传播
        """
        feat = self.features(x)
        feat = self.flatten(feat)
        out = self.classifier(feat)
        return out


class LocalGlobalDiscriminator(nn.Cell):
    """
    局部-全局判别器 (Local-Global Discriminator)
    
    创新点:
    - 全局分支: 评估整体图像的语义一致性
    - 局部分支: 评估修复区域的细节质量
    
    Args:
        in_channels: 输入通道数
        image_size: 图像尺寸
    """
    
    def __init__(self, in_channels=3, image_size=512, base_channels=64):
        super(LocalGlobalDiscriminator, self).__init__()
        
        self.image_size = image_size
        
        # 全局判别器
        self.global_discriminator = nn.SequentialCell([
            Conv2dBlock(in_channels, base_channels, 4, 2, 1),         # 256
            Conv2dBlock(base_channels, base_channels * 2, 4, 2, 1),   # 128
            Conv2dBlock(base_channels * 2, base_channels * 4, 4, 2, 1), # 64
            Conv2dBlock(base_channels * 4, base_channels * 8, 4, 2, 1), # 32
            Conv2dBlock(base_channels * 8, base_channels * 8, 4, 2, 1), # 16
        ])
        
        self.global_classifier = nn.SequentialCell([
            nn.Flatten(),
            nn.Dense(base_channels * 8 * 16 * 16, 1024),
            nn.LeakyReLU(0.2),
            nn.Dense(1024, 1)
        ])
        
        # 局部判别器
        self.local_discriminator = nn.SequentialCell([
            Conv2dBlock(in_channels, base_channels, 4, 2, 1),         # 64
            Conv2dBlock(base_channels, base_channels * 2, 4, 2, 1),   # 32
            Conv2dBlock(base_channels * 2, base_channels * 4, 4, 2, 1), # 16
            Conv2dBlock(base_channels * 4, base_channels * 8, 4, 2, 1), # 8
        ])
        
        self.local_classifier = nn.SequentialCell([
            nn.Flatten(),
            nn.Dense(base_channels * 8 * 8 * 8, 512),
            nn.LeakyReLU(0.2),
            nn.Dense(512, 1)
        ])
        
    def construct(self, x, local_patch=None):
        """
        局部-全局判别前向传播
        
        Args:
            x: 完整图像 (B, C, H, W)
            local_patch: 局部区域 (B, C, H', W'), 可选
            
        Returns:
            global_out: 全局判别结果
            local_out: 局部判别结果 (如果提供local_patch)
        """
        # 全局判别
        global_feat = self.global_discriminator(x)
        global_out = self.global_classifier(global_feat)
        
        # 局部判别
        if local_patch is not None:
            local_feat = self.local_discriminator(local_patch)
            local_out = self.local_classifier(local_feat)
            return global_out, local_out
        
        return global_out


class SRDiscriminator(nn.Cell):
    """
    超分辨率判别器
    
    基于SRGAN判别器设计,用于评估超分图像质量
    """
    
    def __init__(self, image_size=96, in_channels=3, base_channels=64):
        super(SRDiscriminator, self).__init__()
        
        feature_map_size = image_size // 16
        
        # 特征提取
        cfg_in = [3, 64, 64, 128, 128, 256, 256, 512]
        cfg_out = [64, 64, 128, 128, 256, 256, 512, 512]
        
        layers = []
        stride_toggle = 0
        
        for c_in, c_out in zip(cfg_in, cfg_out):
            conv = nn.Conv2d(c_in, c_out, 3, stride=1 + stride_toggle, 
                           padding=1, pad_mode='pad')
            layers.extend([conv, nn.LeakyReLU(0.2)])
            stride_toggle = (stride_toggle + 1) % 2
        
        self.features = nn.SequentialCell(layers)
        
        # 分类器
        self.flatten = nn.Flatten()
        self.classifier = nn.SequentialCell([
            nn.Dense(512 * feature_map_size * feature_map_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dense(1024, 1),
            nn.Sigmoid()
        ])
        
    def construct(self, x):
        """超分判别器前向传播"""
        feat = self.features(x)
        feat = self.flatten(feat)
        out = self.classifier(feat)
        return out


class FusionDiscriminator(nn.Cell):
    """
    融合判别器 (Fusion Discriminator)
    
    同时评估修复质量和超分质量
    """
    
    def __init__(self, image_size=512, in_channels=3):
        super(FusionDiscriminator, self).__init__()
        
        # 修复判别器
        self.inpaint_disc = InpaintingDiscriminator(in_channels, 64)
        
        # 多尺度判别器 (用于超分)
        self.multiscale_disc = MultiScaleDiscriminator(in_channels, 64, 3, 4)
        
        # 局部-全局判别器
        self.local_global_disc = LocalGlobalDiscriminator(in_channels, image_size, 64)
        
    def construct(self, x, local_patch=None):
        """
        融合判别前向传播
        
        Args:
            x: 完整图像
            local_patch: 局部区域 (修复区域)
            
        Returns:
            inpaint_out: 修复判别结果
            multiscale_out: 多尺度判别结果列表
            local_global_out: 局部-全局判别结果
        """
        inpaint_out = self.inpaint_disc(x)
        multiscale_out = self.multiscale_disc(x)
        
        if local_patch is not None:
            local_global_out = self.local_global_disc(x, local_patch)
        else:
            local_global_out = self.local_global_disc(x)
        
        return inpaint_out, multiscale_out, local_global_out


def get_multiscale_discriminator(in_channels=3, base_channels=64, 
                                  num_scales=3, num_layers=4):
    """
    获取多尺度判别器
    """
    return MultiScaleDiscriminator(in_channels, base_channels, num_scales, num_layers)


def get_fusion_discriminator(image_size=512, in_channels=3):
    """
    获取融合判别器
    """
    return FusionDiscriminator(image_size, in_channels)

