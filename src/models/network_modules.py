# Copyright 2024
# CRA-SRGAN Fusion Model - Network Modules
"""
基础网络模块定义
包含门控卷积、残差块、子像素卷积等核心组件
"""

import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import TruncatedNormal, HeNormal
import numpy as np


class GatedConv2d(nn.Cell):
    """
    轻量级门控卷积 (Lightweight Gated Convolution)
    
    通过门控机制自适应处理破损区域,避免普通卷积对无效区域的错误响应
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        dilation: 膨胀率
        use_single_channel: 是否使用单通道门控(LWGCsc)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 dilation=1, use_single_channel=False, use_spectral_norm=False):
        super(GatedConv2d, self).__init__()
        
        self.activation = nn.ELU(alpha=1.0)
        
        # 特征卷积分支
        self.feature_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            pad_mode='same', padding=0, dilation=dilation, 
            has_bias=True, weight_init=TruncatedNormal(0.05)
        )
        
        # 门控分支
        if use_single_channel:
            # LWGCsc: 单通道门控
            self.gate_conv = nn.Conv2d(
                in_channels, 1, kernel_size, stride,
                pad_mode='same', padding=0, dilation=dilation,
                has_bias=True, weight_init=TruncatedNormal(0.05)
            )
        else:
            # LWGCds: 深度可分离门控
            self.gate_conv = nn.Conv2d(
                in_channels, out_channels, 1, stride,
                pad_mode='same', padding=0, dilation=dilation,
                has_bias=True, weight_init=TruncatedNormal(0.05)
            )
        
        self.sigmoid = nn.Sigmoid()
        
    def construct(self, x):
        """前向传播"""
        features = self.feature_conv(x)
        gate = self.gate_conv(x)
        return self.sigmoid(gate) * self.activation(features)


class TransposeGatedConv2d(nn.Cell):
    """
    转置门控卷积 (上采样)
    
    结合双线性插值上采样和门控卷积
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, use_single_channel=False, scale_factor=2):
        super(TransposeGatedConv2d, self).__init__()
        
        self.scale_factor = scale_factor
        self.gated_conv = GatedConv2d(
            in_channels, out_channels, kernel_size, stride,
            dilation, use_single_channel
        )
        
    def construct(self, x):
        """前向传播"""
        shape = x.shape
        new_h = shape[2] * self.scale_factor
        new_w = shape[3] * self.scale_factor
        
        resize_op = ops.ResizeBilinearV2()
        x = resize_op(x, (new_h, new_w))
        x = self.gated_conv(x)
        return x


class ResidualBlock(nn.Cell):
    """
    残差块 (Residual Block)
    
    用于SRGAN生成器的特征提取
    
    Args:
        channels: 通道数
        use_batch_norm: 是否使用批归一化
    """
    
    def __init__(self, channels, use_batch_norm=True):
        super(ResidualBlock, self).__init__()
        
        layers = [
            nn.Conv2d(channels, channels, 3, 1, padding=1, 
                     pad_mode='pad', has_bias=True),
        ]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(channels))
        
        layers.append(nn.PReLU(channels))
        layers.append(nn.Conv2d(channels, channels, 3, 1, padding=1,
                               pad_mode='pad', has_bias=True))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(channels))
        
        self.block = nn.SequentialCell(layers)
        
    def construct(self, x):
        """前向传播"""
        return x + self.block(x)


class SubpixelConvolutionLayer(nn.Cell):
    """
    子像素卷积层 (PixelShuffle)
    
    高效的上采样方式,将通道维度重排为空间维度
    
    Args:
        in_channels: 输入通道数
        scale_factor: 上采样倍数
    """
    
    def __init__(self, in_channels, scale_factor=2):
        super(SubpixelConvolutionLayer, self).__init__()
        
        out_channels = in_channels * (scale_factor ** 2)
        self.conv = nn.Conv2d(
            in_channels, out_channels, 3, 1, padding=1,
            pad_mode='pad', has_bias=True
        )
        self.pixel_shuffle = ops.DepthToSpace(scale_factor)
        self.prelu = nn.PReLU(in_channels)
        
    def construct(self, x):
        """前向传播"""
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class DenseBlock(nn.Cell):
    """
    密集连接块 (Dense Block)
    
    增强特征复用,适用于修复任务中的细节恢复
    """
    
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(DenseBlock, self).__init__()
        
        self.layers = nn.CellList()
        for i in range(num_layers):
            layer = nn.SequentialCell([
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, 1,
                         padding=1, pad_mode='pad', has_bias=True),
                nn.LeakyReLU(0.2)
            ])
            self.layers.append(layer)
        
        self.concat = ops.Concat(axis=1)
        
    def construct(self, x):
        """前向传播"""
        features = [x]
        for layer in self.layers:
            out = layer(self.concat(features))
            features.append(out)
        return self.concat(features)


class ChannelAttention(nn.Cell):
    """
    通道注意力模块 (Channel Attention Module)
    
    学习通道间的依赖关系,增强重要特征通道
    """
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.max_pool = ops.ReduceMax(keep_dims=True)
        
        self.fc = nn.SequentialCell([
            nn.Dense(channels, channels // reduction, has_bias=False),
            nn.ReLU(),
            nn.Dense(channels // reduction, channels, has_bias=False)
        ])
        
        self.sigmoid = nn.Sigmoid()
        self.reshape = ops.Reshape()
        
    def construct(self, x):
        """前向传播"""
        b, c, h, w = x.shape
        
        # 平均池化
        avg_out = self.avg_pool(x, (2, 3))
        avg_out = self.reshape(avg_out, (b, c))
        avg_out = self.fc(avg_out)
        
        # 最大池化
        max_out = self.max_pool(x, (2, 3))
        max_out = self.reshape(max_out, (b, c))
        max_out = self.fc(max_out)
        
        # 融合
        out = self.sigmoid(avg_out + max_out)
        out = self.reshape(out, (b, c, 1, 1))
        
        return x * out


class SpatialAttention(nn.Cell):
    """
    空间注意力模块 (Spatial Attention Module)
    
    学习空间位置间的依赖关系,聚焦于重要空间区域
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding=kernel_size // 2,
                             pad_mode='pad', has_bias=False)
        self.sigmoid = nn.Sigmoid()
        self.concat = ops.Concat(axis=1)
        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.reduce_max = ops.ReduceMax(keep_dims=True)
        
    def construct(self, x):
        """前向传播"""
        avg_out = self.reduce_mean(x, 1)
        max_out = self.reduce_max(x, 1)
        
        attention = self.concat([avg_out, max_out])
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        return x * attention


class CBAM(nn.Cell):
    """
    卷积块注意力模块 (Convolutional Block Attention Module)
    
    结合通道注意力和空间注意力
    """
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def construct(self, x):
        """前向传播"""
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SpectralNormConv2d(nn.Cell):
    """
    谱归一化卷积层 (Spectral Normalization)
    
    稳定GAN训练,防止判别器过强
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, bias=True):
        super(SpectralNormConv2d, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=padding, pad_mode='pad', dilation=dilation, has_bias=bias
        )
        
        # 初始化u和v向量用于谱归一化
        self.u = mindspore.Parameter(
            Tensor(np.random.normal(0, 1, (1, out_channels)), mindspore.float32),
            requires_grad=False
        )
        
    def construct(self, x):
        """前向传播"""
        return self.conv(x)


class CoordinateAttention(nn.Cell):
    """
    坐标注意力模块 (Coordinate Attention)
    
    捕获长程依赖和精确的位置信息
    """
    
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        
        self.pool_h = ops.ReduceMean(keep_dims=True)
        self.pool_w = ops.ReduceMean(keep_dims=True)
        
        mid_channels = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, has_bias=True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.HSwish()
        
        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1, has_bias=True)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1, has_bias=True)
        
        self.sigmoid = nn.Sigmoid()
        self.transpose = ops.Transpose()
        self.concat = ops.Concat(axis=2)
        
    def construct(self, x):
        """前向传播"""
        b, c, h, w = x.shape
        
        # 水平和垂直方向的池化
        x_h = self.pool_h(x, 3)  # (b, c, h, 1)
        x_w = self.pool_w(x, 2)  # (b, c, 1, w)
        x_w = self.transpose(x_w, (0, 1, 3, 2))  # (b, c, w, 1)
        
        # 拼接并通过共享卷积
        y = self.concat([x_h, x_w])  # (b, c, h+w, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # 分割并生成注意力权重
        split_op = ops.Split(axis=2, output_num=2)
        x_h, x_w = split_op(y)
        x_w = self.transpose(x_w, (0, 1, 3, 2))
        
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        
        return x * a_h * a_w


class FrequencyDecompositionModule(nn.Cell):
    """
    频率分解模块 (Frequency Decomposition Module)
    
    创新点: 将图像分解为低频(结构)和高频(细节)成分分别处理
    用于修复任务中的结构恢复和细节增强
    """
    
    def __init__(self, channels):
        super(FrequencyDecompositionModule, self).__init__()
        
        # 低频分支 - 大感受野,捕获结构信息
        self.low_freq_branch = nn.SequentialCell([
            nn.Conv2d(channels, channels, 7, 1, padding=3, pad_mode='pad'),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 5, 1, padding=2, pad_mode='pad'),
            nn.BatchNorm2d(channels)
        ])
        
        # 高频分支 - 小感受野,捕获细节信息
        self.high_freq_branch = nn.SequentialCell([
            nn.Conv2d(channels, channels, 3, 1, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1, 1),
            nn.BatchNorm2d(channels)
        ])
        
        # 融合卷积
        self.fusion_conv = nn.Conv2d(channels * 2, channels, 1, 1)
        
        self.concat = ops.Concat(axis=1)
        
    def construct(self, x):
        """前向传播"""
        low_freq = self.low_freq_branch(x)
        high_freq = x - low_freq  # 高频 = 原始 - 低频
        high_freq = self.high_freq_branch(high_freq)
        
        # 融合低频和增强的高频
        out = self.concat([low_freq, high_freq])
        out = self.fusion_conv(out)
        
        return out + x  # 残差连接


class EdgeEnhancementModule(nn.Cell):
    """
    边缘增强模块 (Edge Enhancement Module)
    
    创新点: 显式提取和增强边缘信息,提升修复边界的清晰度
    """
    
    def __init__(self, channels):
        super(EdgeEnhancementModule, self).__init__()
        
        # Sobel边缘检测算子
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        # 扩展到所有通道
        self.sobel_x = Tensor(np.stack([sobel_x] * channels, axis=0).reshape(channels, 1, 3, 3))
        self.sobel_y = Tensor(np.stack([sobel_y] * channels, axis=0).reshape(channels, 1, 3, 3))
        
        # 边缘特征处理
        self.edge_conv = nn.SequentialCell([
            nn.Conv2d(channels * 2, channels, 3, 1, padding=1, pad_mode='pad'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, 1, padding=1, pad_mode='pad')
        ])
        
        # 边缘注意力
        self.edge_attention = nn.Conv2d(channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.concat = ops.Concat(axis=1)
        
    def construct(self, x):
        """前向传播"""
        # 边缘检测 (简化实现,使用可学习卷积替代固定Sobel)
        edge_x = ops.conv2d(x, self.sobel_x, pad_mode='same', groups=x.shape[1])
        edge_y = ops.conv2d(x, self.sobel_y, pad_mode='same', groups=x.shape[1])
        
        # 边缘幅度
        edge_magnitude = ops.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        
        # 边缘特征增强
        edge_features = self.concat([edge_magnitude, x])
        edge_enhanced = self.edge_conv(edge_features)
        
        # 边缘注意力加权
        edge_attention = self.sigmoid(self.edge_attention(edge_enhanced))
        
        return x + edge_enhanced * edge_attention

