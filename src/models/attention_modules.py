# Copyright 2024
# CRA-SRGAN Fusion Model - Attention Modules
"""
多尺度上下文注意力模块

核心创新: 
1. 多尺度上下文注意力 (Multi-Scale Contextual Attention)
2. 跨模态融合注意力 (Cross-Modal Fusion Attention)  
3. 边缘感知注意力 (Edge-Aware Attention)
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import TruncatedNormal
import numpy as np


def downsample(x, factor=2):
    """
    下采样函数
    
    Args:
        x: 输入张量 (B, C, H, W)
        factor: 下采样因子
    """
    shp = x.shape
    unfold = nn.Unfold([1, 1, 1, 1], [1, factor, factor, 1], [1, 1, 1, 1], 'same')
    x = unfold(x)
    reshape = ops.Reshape()
    x = reshape(x, (shp[0], shp[1], shp[2] // factor, shp[3] // factor))
    return x


class InitConv2d(nn.Cell):
    """
    可动态赋权重的卷积层
    用于注意力计算中的patch匹配
    """
    
    def __init__(self, shape, rate=1, is_conv=True):
        super(InitConv2d, self).__init__()
        self.shape = shape
        self.rate = rate
        self.is_conv = is_conv
        
        h, w, i, o = shape
        
        if is_conv:
            self.conv = nn.Conv2d(i, o, (h, w), (1, 1), pad_mode='same')
        else:
            self.conv = nn.Conv2dTranspose(o, i, (h, w), (rate, rate), pad_mode='same')
        
        # 冻结参数
        for param in self.conv.get_parameters():
            param.requires_grad = False
            
        self.params = mindspore.ParameterTuple(self.get_parameters())
        
    def construct(self, x, weight):
        """前向传播"""
        for param in self.params:
            ops.Assign()(param, weight)
        return self.conv(x)


class MultiScaleContextualAttention(nn.Cell):
    """
    多尺度上下文注意力模块 (Multi-Scale Contextual Attention)
    
    创新点: 
    - 在多个尺度上计算注意力分数,捕获不同粒度的上下文信息
    - 自适应权重融合不同尺度的注意力结果
    - 注意力分数共享机制,减少计算量
    
    Args:
        softmax_scale: softmax缩放因子
        num_scales: 注意力尺度数量
        fuse: 是否使用分数融合
    """
    
    def __init__(self, softmax_scale=10, num_scales=3, fuse=True, 
                 dtype=mindspore.float32):
        super(MultiScaleContextualAttention, self).__init__()
        
        self.softmax_scale = softmax_scale
        self.num_scales = num_scales
        self.fuse = fuse
        self.dtype = dtype
        
        # 基础操作
        self.reduce_sum = ops.ReduceSum(False)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.concat = ops.Concat(0)
        self.stack = ops.Stack(axis=0)
        
        # 池化操作
        self.pool1 = nn.MaxPool2d(16, 16, pad_mode='same')
        self.pool2 = nn.MaxPool2d(3, 1, pad_mode='same')
        
        # 数学操作
        self.maximum = ops.Maximum()
        self.sqrt = ops.Sqrt()
        self.square = ops.Square()
        self.softmax = nn.Softmax(axis=1)
        
        # 多尺度Unfold操作 (不同的patch大小)
        self.unfold_ops = nn.CellList([
            nn.Unfold([1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'same'),  # 3x3
            nn.Unfold([1, 5, 5, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'same'),  # 5x5
            nn.Unfold([1, 7, 7, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'same'),  # 7x7
        ][:num_scales])
        
        # 原始特征提取 (用于上采样)
        self.raw_unfold = nn.Unfold([1, 3, 3, 1], [1, 2, 2, 1], [1, 1, 1, 1], 'same')
        
        # 可学习的尺度权重
        self.scale_weights = mindspore.Parameter(
            Tensor(np.ones(num_scales) / num_scales, mindspore.float32),
            name="scale_weights"
        )
        
        # 动态卷积层
        self.conv1 = InitConv2d([3, 3, 128, 1024], 1, True)
        self.conv2 = InitConv2d([3, 3, 1, 1], 1, True)
        self.deconv = InitConv2d([3, 3, 128, 1024], 2, False)
        
        # 尺度融合注意力
        self.fusion_attention = nn.SequentialCell([
            nn.Conv2d(128 * num_scales, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, num_scales, 1),
            nn.Softmax(axis=1)
        ])
        
    def _compute_attention_single_scale(self, src, ref, mask, unfold_op, method='SOFT'):
        """
        计算单一尺度的注意力分数
        
        Args:
            src: 前景特征 (待填充区域)
            ref: 背景特征 (参考区域)
            mask: 掩码
            unfold_op: 展开操作
            method: 注意力方法 ('SOFT' or 'HARD')
        """
        shape_src = src.shape
        batch_size = shape_src[0]
        nc = shape_src[1]
        
        # 提取原始特征patches
        raw_feats = self.raw_unfold(ref)
        raw_feats = self.transpose(raw_feats, (0, 2, 3, 1))
        raw_feats = self.reshape(raw_feats, (batch_size, -1, 3, 3, nc))
        raw_feats = self.transpose(raw_feats, (0, 2, 3, 4, 1))
        
        # 下采样
        src_down = downsample(src)
        ref_down = downsample(ref)
        
        ss = src_down.shape
        rs = ref_down.shape
        
        # 提取参考patches
        feats = unfold_op(ref_down)
        feats = self.transpose(feats, (0, 2, 3, 1))
        
        # 处理mask
        mask_processed = self.pool1(mask)
        mask_processed = self.pool2(mask_processed)
        mask_processed = 1 - mask_processed
        mask_processed = self.reshape(mask_processed, (1, -1, 1, 1))
        
        y_lst = []
        y_up_lst = []
        
        # 分batch处理
        split = ops.Split(0, batch_size)
        src_lst = split(src_down)
        raw_feats_lst = split(raw_feats)
        
        # 构建fuse权重
        fuse_weight = self.reshape(ops.Eye()(3, 3, mindspore.float32), (3, 3, 1, 1))
        
        for idx in range(batch_size):
            x = src_lst[idx]
            raw_r = raw_feats_lst[idx]
            
            # 计算相似度
            r = feats[idx:idx+1]
            r = self.reshape(r, (-1, 3, 3, nc))
            r = self.transpose(r, (0, 2, 3, 1))  # 调整维度顺序
            
            # 归一化
            r_norm = r / self.maximum(self.sqrt(self.reduce_sum(self.square(r), [0, 1, 2])), 1e-8)
            r_kernel = self.transpose(r_norm, (3, 2, 0, 1))
            
            # 计算相关性
            y = self.conv1(x, r_kernel)
            
            if self.fuse:
                # 融合相邻分数
                yi = self.reshape(y, (1, 1, ss[2] * ss[3], rs[2] * rs[3]))
                fuse_kernel = self.transpose(fuse_weight, (3, 2, 0, 1))
                yi = self.conv2(yi, fuse_kernel)
                yi = self.transpose(yi, (0, 2, 3, 1))
                yi = self.reshape(yi, (1, ss[2], ss[3], rs[2], rs[3]))
                yi = self.transpose(yi, (0, 2, 1, 4, 3))
                yi = self.reshape(yi, (1, ss[2] * ss[3], rs[2] * rs[3], 1))
                yi = self.transpose(yi, (0, 3, 1, 2))
                yi = self.conv2(yi, fuse_kernel)
                yi = self.transpose(yi, (0, 2, 3, 1))
                yi = self.reshape(yi, (1, ss[3], ss[2], rs[3], rs[2]))
                y = self.transpose(yi, (0, 2, 1, 4, 3))
            
            # 注意力计算
            y = self.reshape(y, (1, ss[2], ss[3], rs[2] * rs[3]))
            y = self.transpose(y, (0, 3, 1, 2))
            
            if method == 'SOFT':
                y = self.softmax(y * mask_processed * self.softmax_scale) * mask_processed
            
            y = self.reshape(y, (1, rs[2] * rs[3], ss[2], ss[3]))
            
            # 上采样重建
            feats_raw = raw_r[0]
            feats_kernel = self.transpose(feats_raw, (3, 2, 0, 1))
            y_up = self.deconv(y, feats_kernel)
            
            y_lst.append(y)
            y_up_lst.append(y_up)
        
        out = self.concat(y_up_lst)
        correspondence = self.concat(y_lst)
        out = self.reshape(out, (shape_src[0], shape_src[1], shape_src[2], shape_src[3]))
        
        return out, correspondence
        
    def construct(self, src, ref, mask, method='SOFT'):
        """
        多尺度注意力前向传播
        
        Args:
            src: 前景特征 (B, C, H, W)
            ref: 背景特征 (B, C, H, W)
            mask: 掩码 (B, 1, H, W)
            method: 注意力方法
            
        Returns:
            out: 填充后的特征
            correspondence: 注意力分数
        """
        multi_scale_outs = []
        multi_scale_correspondences = []
        
        # 计算各尺度的注意力
        for i, unfold_op in enumerate(self.unfold_ops):
            out, corr = self._compute_attention_single_scale(
                src, ref, mask, unfold_op, method
            )
            multi_scale_outs.append(out)
            multi_scale_correspondences.append(corr)
        
        # 自适应融合
        weights = ops.Softmax(axis=0)(self.scale_weights)
        
        combined_out = multi_scale_outs[0] * weights[0]
        combined_corr = multi_scale_correspondences[0] * weights[0]
        
        for i in range(1, self.num_scales):
            combined_out = combined_out + multi_scale_outs[i] * weights[i]
            combined_corr = combined_corr + multi_scale_correspondences[i] * weights[i]
        
        return combined_out, combined_corr


class CrossModalFusionAttention(nn.Cell):
    """
    跨模态融合注意力 (Cross-Modal Fusion Attention)
    
    创新点: 融合修复分支和超分分支的特征,实现信息互补
    - 修复分支提供语义完整性
    - 超分分支提供高频细节
    
    Args:
        inpaint_channels: 修复分支通道数
        sr_channels: 超分分支通道数
        hidden_channels: 隐藏层通道数
    """
    
    def __init__(self, inpaint_channels, sr_channels, hidden_channels=256, num_heads=8):
        super(CrossModalFusionAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 投影层
        self.inpaint_proj = nn.Conv2d(inpaint_channels, hidden_channels, 1)
        self.sr_proj = nn.Conv2d(sr_channels, hidden_channels, 1)
        
        # Query, Key, Value
        self.q_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.k_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.v_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)
        
        # 输出投影
        self.out_proj = nn.Conv2d(hidden_channels, hidden_channels, 1)
        
        # 门控融合
        self.gate = nn.SequentialCell([
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1),
            nn.Sigmoid()
        ])
        
        # 最终融合
        self.fusion_conv = nn.SequentialCell([
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, pad_mode='pad')
        ])
        
        self.softmax = nn.Softmax(axis=-1)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.concat = ops.Concat(axis=1)
        self.bmm = ops.BatchMatMul()
        
    def construct(self, inpaint_feat, sr_feat):
        """
        跨模态注意力前向传播
        
        Args:
            inpaint_feat: 修复分支特征 (B, C1, H, W)
            sr_feat: 超分分支特征 (B, C2, H, W)
            
        Returns:
            fused_feat: 融合后的特征
        """
        b, _, h, w = inpaint_feat.shape
        
        # 投影到相同维度
        inpaint_proj = self.inpaint_proj(inpaint_feat)
        sr_proj = self.sr_proj(sr_feat)
        
        # 计算Q, K, V
        # 修复特征作为Query, 超分特征提供Key和Value
        q = self.q_proj(inpaint_proj)
        k = self.k_proj(sr_proj)
        v = self.v_proj(sr_proj)
        
        # 多头注意力
        c = q.shape[1]
        
        q = self.reshape(q, (b, self.num_heads, self.head_dim, h * w))
        k = self.reshape(k, (b, self.num_heads, self.head_dim, h * w))
        v = self.reshape(v, (b, self.num_heads, self.head_dim, h * w))
        
        q = self.transpose(q, (0, 1, 3, 2))  # (B, heads, HW, dim)
        k = self.transpose(k, (0, 1, 2, 3))  # (B, heads, dim, HW)
        
        # 注意力分数
        attn = self.bmm(q, k) * self.scale
        attn = self.softmax(attn)
        
        # 应用注意力
        v = self.transpose(v, (0, 1, 3, 2))  # (B, heads, HW, dim)
        out = self.bmm(attn, v)  # (B, heads, HW, dim)
        
        out = self.transpose(out, (0, 1, 3, 2))  # (B, heads, dim, HW)
        out = self.reshape(out, (b, c, h, w))
        out = self.out_proj(out)
        
        # 门控融合
        gate_input = self.concat([inpaint_proj, out])
        gate_weight = self.gate(gate_input)
        
        gated_out = gate_weight * out + (1 - gate_weight) * inpaint_proj
        
        # 最终融合
        fusion_input = self.concat([gated_out, sr_proj])
        fused_feat = self.fusion_conv(fusion_input)
        
        return fused_feat


class EdgeAwareAttention(nn.Cell):
    """
    边缘感知注意力 (Edge-Aware Attention)
    
    创新点: 在注意力计算中显式引入边缘信息
    - 增强修复边界的连贯性
    - 保护重要的纹理细节
    
    Args:
        channels: 输入通道数
    """
    
    def __init__(self, channels, reduction=8):
        super(EdgeAwareAttention, self).__init__()
        
        # 边缘检测分支
        self.edge_conv1 = nn.Conv2d(channels, channels // 2, 3, padding=1, pad_mode='pad')
        self.edge_conv2 = nn.Conv2d(channels // 2, 1, 3, padding=1, pad_mode='pad')
        
        # 特征注意力分支
        self.feat_conv1 = nn.Conv2d(channels, channels // reduction, 1)
        self.feat_conv2 = nn.Conv2d(channels // reduction, channels, 1)
        
        # 边缘引导的注意力
        self.edge_attention = nn.SequentialCell([
            nn.Conv2d(channels + 1, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        ])
        
        # 输出卷积
        self.output_conv = nn.Conv2d(channels * 2, channels, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.concat = ops.Concat(axis=1)
        
    def construct(self, x):
        """
        边缘感知注意力前向传播
        
        Args:
            x: 输入特征 (B, C, H, W)
            
        Returns:
            out: 边缘增强的特征
        """
        # 提取边缘特征
        edge = self.relu(self.edge_conv1(x))
        edge = self.sigmoid(self.edge_conv2(edge))
        
        # 特征注意力
        feat_attn = self.relu(self.feat_conv1(x))
        feat_attn = self.sigmoid(self.feat_conv2(feat_attn))
        
        # 边缘引导的注意力
        edge_guided = self.concat([x, edge])
        edge_attn = self.edge_attention(edge_guided)
        
        # 融合
        attended_feat = x * feat_attn + x * edge_attn
        output = self.concat([x, attended_feat])
        output = self.output_conv(output)
        
        return output


class ApplyMultiScaleAttention(nn.Cell):
    """
    多尺度注意力转移模块 (Multi-Scale Attention Transfer)
    
    将注意力分数应用到不同分辨率的特征图
    """
    
    def __init__(self, feature_shape, attention_shape):
        super(ApplyMultiScaleAttention, self).__init__()
        
        self.feature_shape = feature_shape  # [B, C, H, W]
        self.attention_shape = attention_shape  # [B, N, H', W']
        
        self.rate = feature_shape[2] // attention_shape[2]
        self.kernel_size = self.rate * 2
        
        # Unfold操作
        self.unfold = nn.Unfold(
            [1, self.kernel_size, self.kernel_size, 1],
            [1, self.rate, self.rate, 1],
            [1, 1, 1, 1], 'same'
        )
        
        # 反卷积
        self.deconv = InitConv2d(
            [self.kernel_size, self.kernel_size, feature_shape[1], 1024],
            self.rate, False
        )
        
        # 后处理卷积
        from .network_modules import GatedConv2d
        self.post_conv = nn.SequentialCell([
            GatedConv2d(feature_shape[1], feature_shape[1], 3, 1, 1),
            GatedConv2d(feature_shape[1], feature_shape[1], 3, 1, 2)
        ])
        
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.concat = ops.Concat(0)
        
    def construct(self, x, correspondence):
        """
        应用注意力前向传播
        
        Args:
            x: 输入特征 (B, C, H, W)
            correspondence: 注意力分数
            
        Returns:
            out: 注意力转移后的特征
        """
        batch_size = self.feature_shape[0]
        nc = self.feature_shape[1]
        
        # 展开特征
        raw_feats = self.unfold(x)
        raw_feats = self.transpose(raw_feats, (0, 2, 3, 1))
        raw_feats = self.reshape(raw_feats, 
            (batch_size, -1, self.kernel_size, self.kernel_size, nc))
        raw_feats = self.transpose(raw_feats, (0, 2, 3, 4, 1))
        
        # 应用注意力
        split = ops.Split(0, batch_size)
        raw_feats_lst = split(raw_feats)
        correspondence = self.transpose(correspondence, (0, 2, 3, 1))
        att_lst = split(correspondence)
        
        outputs = []
        for feats, att in zip(raw_feats_lst, att_lst):
            feats_kernel = self.transpose(feats[0], (3, 2, 0, 1))
            att = self.transpose(att, (0, 3, 1, 2))
            y = self.deconv(att, feats_kernel)
            outputs.append(y)
        
        out = self.concat(outputs)
        out = self.post_conv(out)
        
        return out


class ProgressiveAttention(nn.Cell):
    """
    渐进式注意力模块 (Progressive Attention)
    
    创新点: 从粗到细逐步细化注意力
    适用于渐进式超分辨率重建
    """
    
    def __init__(self, channels_list, num_stages=4):
        super(ProgressiveAttention, self).__init__()
        
        self.num_stages = num_stages
        self.stages = nn.CellList()
        
        for i, channels in enumerate(channels_list):
            stage = nn.SequentialCell([
                nn.Conv2d(channels, channels, 3, padding=1, pad_mode='pad'),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 1),
                nn.Sigmoid()
            ])
            self.stages.append(stage)
        
        # 尺度间的特征传递
        self.upsample = ops.ResizeBilinearV2()
        
    def construct(self, features_list):
        """
        渐进式注意力前向传播
        
        Args:
            features_list: 多尺度特征列表 [(B,C1,H1,W1), (B,C2,H2,W2), ...]
            
        Returns:
            attended_features: 注意力增强的特征列表
        """
        attended_features = []
        prev_attention = None
        
        for i, (feat, stage) in enumerate(zip(features_list, self.stages)):
            # 计算当前尺度的注意力
            attention = stage(feat)
            
            # 如果存在前一尺度的注意力,进行融合
            if prev_attention is not None:
                # 上采样前一尺度的注意力
                h, w = feat.shape[2], feat.shape[3]
                upsampled_attention = self.upsample(prev_attention, (h, w))
                
                # 融合 (取最大值)
                attention = ops.Maximum()(attention, upsampled_attention)
            
            attended_feat = feat * attention
            attended_features.append(attended_feat)
            prev_attention = attention
        
        return attended_features

