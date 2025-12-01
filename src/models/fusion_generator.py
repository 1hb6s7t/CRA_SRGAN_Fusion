# Copyright 2024
# CRA-SRGAN Fusion Generator
"""
图像修复与超高清化一体化生成器

核心创新:
1. 双流编码器架构 - 修复流+超分流
2. 多尺度上下文残差聚合 - 增强破损区域的语义完整性
3. 渐进式上采样模块 - 从512到8K的逐步重建
4. 边缘感知细节保真 - 保护纹理和边缘细节
"""

import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import TruncatedNormal, HeNormal
import numpy as np

from .network_modules import (
    GatedConv2d, TransposeGatedConv2d, ResidualBlock,
    SubpixelConvolutionLayer, DenseBlock, CBAM,
    FrequencyDecompositionModule, EdgeEnhancementModule, CoordinateAttention
)
from .attention_modules import (
    MultiScaleContextualAttention, CrossModalFusionAttention,
    EdgeAwareAttention, ApplyMultiScaleAttention, ProgressiveAttention
)


class InpaintingEncoder(nn.Cell):
    """
    修复编码器 (Inpainting Encoder)
    
    使用门控卷积处理破损区域,避免无效信息传播
    """
    
    def __init__(self, in_channels=4, base_channels=32):
        super(InpaintingEncoder, self).__init__()
        
        # 编码器层 (从512x512到64x64)
        self.enc1 = nn.SequentialCell([
            GatedConv2d(in_channels, base_channels, 5, 2, 1, use_single_channel=True),
            GatedConv2d(base_channels, base_channels, 3, 1, 1, use_single_channel=True)
        ])  # 256x256
        
        self.enc2 = nn.SequentialCell([
            GatedConv2d(base_channels, base_channels * 2, 3, 2, 1, use_single_channel=True),
            GatedConv2d(base_channels * 2, base_channels * 2, 3, 1, 1, use_single_channel=True)
        ])  # 128x128
        
        self.enc3 = nn.SequentialCell([
            GatedConv2d(base_channels * 2, base_channels * 4, 3, 2, 1, use_single_channel=True),
            GatedConv2d(base_channels * 4, base_channels * 4, 3, 1, 1, use_single_channel=True)
        ])  # 64x64
        
        # 扩张卷积增大感受野
        self.dilated_convs = nn.SequentialCell([
            GatedConv2d(base_channels * 4, base_channels * 4, 3, 1, 2, use_single_channel=True),
            GatedConv2d(base_channels * 4, base_channels * 4, 3, 1, 4, use_single_channel=True),
            GatedConv2d(base_channels * 4, base_channels * 4, 3, 1, 8, use_single_channel=True),
            GatedConv2d(base_channels * 4, base_channels * 4, 3, 1, 16, use_single_channel=True)
        ])
        
    def construct(self, x):
        """
        编码器前向传播
        
        Returns:
            features: 多尺度特征字典
        """
        f1 = self.enc1(x)      # 256x256, 32ch
        f2 = self.enc2(f1)     # 128x128, 64ch
        f3 = self.enc3(f2)     # 64x64, 128ch
        f4 = self.dilated_convs(f3)  # 64x64, 128ch (增大感受野)
        
        return {'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4}


class InpaintingDecoder(nn.Cell):
    """
    修复解码器 (Inpainting Decoder)
    
    结合注意力机制和上采样恢复破损区域
    """
    
    def __init__(self, base_channels=32, batch_size=4):
        super(InpaintingDecoder, self).__init__()
        
        self.batch_size = batch_size
        
        # 解码器层
        self.dec3 = nn.SequentialCell([
            TransposeGatedConv2d(base_channels * 4, base_channels * 2, 3, 1, 1),
            GatedConv2d(base_channels * 2, base_channels * 2, 3, 1, 1)
        ])  # 128x128
        
        self.dec2 = nn.SequentialCell([
            TransposeGatedConv2d(base_channels * 4, base_channels, 3, 1, 1),
            GatedConv2d(base_channels, base_channels, 3, 1, 1)
        ])  # 256x256
        
        self.dec1 = nn.SequentialCell([
            TransposeGatedConv2d(base_channels * 2, base_channels // 2, 3, 1, 1),
            nn.Conv2d(base_channels // 2, 3, 3, pad_mode='same')
        ])  # 512x512
        
        # 注意力转移模块
        self.apply_attention_128 = ApplyMultiScaleAttention(
            [batch_size, 64, 128, 128], [batch_size, 1024, 32, 32]
        )
        self.apply_attention_256 = ApplyMultiScaleAttention(
            [batch_size, 32, 256, 256], [batch_size, 1024, 32, 32]
        )
        
        self.concat = ops.Concat(axis=1)
        
    def construct(self, enc_features, attention_out, correspondence):
        """
        解码器前向传播
        
        Args:
            enc_features: 编码器特征字典
            attention_out: 注意力输出
            correspondence: 注意力分数
        """
        # 融合注意力输出和编码器特征
        f4 = enc_features['f4']
        f3 = enc_features['f3']
        f2 = enc_features['f2']
        f1 = enc_features['f1']
        
        # 解码 64->128
        d3 = self.dec3(attention_out)
        # 应用注意力转移
        att_f2 = self.apply_attention_128(f2, correspondence)
        d3 = self.concat([d3, att_f2])
        
        # 解码 128->256
        d2 = self.dec2(d3)
        att_f1 = self.apply_attention_256(f1, correspondence)
        d2 = self.concat([d2, att_f1])
        
        # 解码 256->512
        d1 = self.dec1(d2)
        
        # 限制输出范围
        out = ops.clip_by_value(d1, -1, 1)
        
        return out


class CoarseNetwork(nn.Cell):
    """
    粗修复网络 (Coarse Network)
    
    第一阶段: 生成粗糙的修复结果,建立全局结构
    """
    
    def __init__(self, in_channels=4, base_channels=32):
        super(CoarseNetwork, self).__init__()
        
        # 编码
        self.encode = nn.SequentialCell([
            GatedConv2d(in_channels, base_channels, 5, 2, 1, use_single_channel=True),
            GatedConv2d(base_channels, base_channels, 3, 1, 1, use_single_channel=True),
            GatedConv2d(base_channels, base_channels * 2, 3, 2, 1, use_single_channel=True)
        ])
        
        # 中间处理
        self.middle = nn.SequentialCell([
            GatedConv2d(base_channels * 2, base_channels * 2, 3, 1, 1, use_single_channel=True),
            GatedConv2d(base_channels * 2, base_channels * 2, 3, 1, 1, use_single_channel=True),
            GatedConv2d(base_channels * 2, base_channels * 2, 3, 1, 2, use_single_channel=True),
            GatedConv2d(base_channels * 2, base_channels * 2, 3, 1, 4, use_single_channel=True),
            GatedConv2d(base_channels * 2, base_channels * 2, 3, 1, 8, use_single_channel=True),
            GatedConv2d(base_channels * 2, base_channels * 2, 3, 1, 1, use_single_channel=True)
        ])
        
        # 解码
        self.decode = nn.SequentialCell([
            TransposeGatedConv2d(base_channels * 2, base_channels, 3, 1, 1, use_single_channel=True),
            GatedConv2d(base_channels, base_channels, 3, 1, 1, use_single_channel=True),
            TransposeGatedConv2d(base_channels, 3, 3, 1, 1, use_single_channel=True)
        ])
        
    def construct(self, x):
        """粗网络前向传播"""
        x = self.encode(x)
        x = self.middle(x)
        x = self.decode(x)
        return ops.clip_by_value(x, -1, 1)


class RefineNetwork(nn.Cell):
    """
    细修复网络 (Refine Network)
    
    第二阶段: 利用注意力机制细化修复结果
    """
    
    def __init__(self, base_channels=32, batch_size=4, attention_type='SOFT'):
        super(RefineNetwork, self).__init__()
        
        self.batch_size = batch_size
        
        # 编码器
        self.encoder = InpaintingEncoder(4, base_channels)
        
        # 注意力模块
        self.contextual_attention = MultiScaleContextualAttention(
            softmax_scale=10, num_scales=3, fuse=True
        )
        
        # 注意力特征融合
        self.attention_conv1 = GatedConv2d(base_channels * 4, base_channels * 4, 3, 1, 1)
        self.attention_conv2 = GatedConv2d(base_channels * 8, base_channels * 4, 3, 1, 1)
        
        # 解码器
        self.decoder = InpaintingDecoder(base_channels, batch_size)
        
        self.concat = ops.Concat(axis=1)
        self.attention_type = attention_type
        
    def construct(self, x, mask):
        """
        细修复网络前向传播
        
        Args:
            x: 输入图像 (包含粗修复结果)
            mask: 破损区域掩码
        """
        # 编码
        enc_features = self.encoder(x)
        f4 = enc_features['f4']
        
        # 计算上下文注意力
        attention_out, correspondence = self.contextual_attention(
            f4, f4, mask, self.attention_type
        )
        
        # 融合注意力特征
        attention_out = self.attention_conv1(attention_out)
        fusion = self.concat([f4, attention_out])
        fusion = self.attention_conv2(fusion)
        
        # 解码
        out = self.decoder(enc_features, fusion, correspondence)
        
        return out, correspondence


class SRBranch(nn.Cell):
    """
    超分辨率分支 (Super-Resolution Branch)
    
    基于SRGAN的残差网络结构,支持渐进式上采样
    """
    
    def __init__(self, upscale_factor=16, base_channels=64, num_res_blocks=16):
        super(SRBranch, self).__init__()
        
        self.upscale_factor = upscale_factor
        num_upsample = int(math.log2(upscale_factor))
        
        # 第一层卷积
        self.conv_first = nn.SequentialCell([
            nn.Conv2d(3, base_channels, 9, 1, padding=4, pad_mode='pad', has_bias=True),
            nn.PReLU(base_channels)
        ])
        
        # 残差块
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(base_channels))
        self.res_blocks = nn.SequentialCell(res_blocks)
        
        # 残差后的卷积
        self.conv_after_res = nn.SequentialCell([
            nn.Conv2d(base_channels, base_channels, 3, 1, padding=1, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(base_channels)
        ])
        
        # 渐进式上采样模块
        upsample_layers = []
        for i in range(num_upsample):
            upsample_layers.append(SubpixelConvolutionLayer(base_channels, scale_factor=2))
        self.upsample = nn.SequentialCell(upsample_layers)
        
        # 输出卷积
        self.conv_last = nn.Conv2d(base_channels, 3, 9, 1, padding=4, pad_mode='pad', has_bias=True)
        
        self.tanh = nn.Tanh()
        
    def construct(self, x):
        """
        超分分支前向传播
        
        Args:
            x: 低分辨率输入 (B, 3, H, W)
            
        Returns:
            out: 高分辨率输出 (B, 3, H*scale, W*scale)
        """
        first = self.conv_first(x)
        res = self.res_blocks(first)
        res = self.conv_after_res(res)
        
        # 残差连接
        out = first + res
        
        # 上采样
        out = self.upsample(out)
        out = self.conv_last(out)
        out = self.tanh(out)
        
        return out


class ProgressiveSRBranch(nn.Cell):
    """
    渐进式超分辨率分支 (Progressive Super-Resolution Branch)
    
    创新点: 分阶段逐步提升分辨率,每个阶段都有监督
    512 -> 1024 -> 2048 -> 4096 -> 8192
    """
    
    def __init__(self, base_channels=64, num_res_blocks_per_stage=4):
        super(ProgressiveSRBranch, self).__init__()
        
        # 初始特征提取
        self.conv_first = nn.SequentialCell([
            nn.Conv2d(3, base_channels, 9, 1, padding=4, pad_mode='pad'),
            nn.PReLU(base_channels)
        ])
        
        # 4个上采样阶段
        self.stages = nn.CellList()
        self.stage_outputs = nn.CellList()
        
        for i in range(4):
            # 每个阶段的残差块
            res_blocks = []
            for _ in range(num_res_blocks_per_stage):
                res_blocks.append(ResidualBlock(base_channels, use_batch_norm=True))
            
            stage = nn.SequentialCell([
                *res_blocks,
                SubpixelConvolutionLayer(base_channels, scale_factor=2)
            ])
            self.stages.append(stage)
            
            # 每个阶段的输出头
            output_head = nn.Conv2d(base_channels, 3, 3, padding=1, pad_mode='pad')
            self.stage_outputs.append(output_head)
        
        self.tanh = nn.Tanh()
        
    def construct(self, x, return_intermediates=False):
        """
        渐进式超分前向传播
        
        Args:
            x: 低分辨率输入 (B, 3, 512, 512)
            return_intermediates: 是否返回中间结果
            
        Returns:
            out: 最终高分辨率输出 (B, 3, 8192, 8192)
            intermediates: 中间阶段输出列表 (可选)
        """
        feat = self.conv_first(x)
        intermediates = []
        
        for i, (stage, output_head) in enumerate(zip(self.stages, self.stage_outputs)):
            feat = stage(feat)
            
            if return_intermediates or i == len(self.stages) - 1:
                out = output_head(feat)
                out = self.tanh(out)
                intermediates.append(out)
        
        if return_intermediates:
            return intermediates[-1], intermediates
        return intermediates[-1]


class CRASRGANGenerator(nn.Cell):
    """
    CRA-SRGAN融合生成器 (主网络)
    
    核心架构:
    1. 粗修复网络 (Coarse Network) - 建立全局结构
    2. 细修复网络 (Refine Network) - 利用上下文注意力细化
    3. 上下文残差聚合 (CRA Module) - 高频细节恢复
    4. 渐进式超分分支 (Progressive SR) - 从512到8K的逐步重建
    5. 跨模态融合 (Cross-Modal Fusion) - 修复与超分信息互补
    
    Args:
        config: 模型配置
    """
    
    def __init__(self, config):
        super(CRASRGANGenerator, self).__init__()
        
        self.config = config
        self.batch_size = config.training.batch_size if hasattr(config, 'training') else 4
        
        # 模型参数
        base_channels = 32
        sr_channels = 64
        upscale_factor = config.model.srgan_upscale_factor if hasattr(config, 'model') else 16
        
        # 1. 粗修复网络
        self.coarse_net = CoarseNetwork(4, base_channels)
        
        # 2. 细修复网络
        self.refine_net = RefineNetwork(base_channels, self.batch_size, 'SOFT')
        
        # 3. 上下文残差聚合模块
        self.cra_module = ContextualResidualAggregation(base_channels * 4)
        
        # 4. 渐进式超分分支
        self.sr_branch = ProgressiveSRBranch(sr_channels, num_res_blocks_per_stage=4)
        
        # 5. 跨模态融合注意力
        self.cross_modal_fusion = CrossModalFusionAttention(
            inpaint_channels=3,
            sr_channels=3,
            hidden_channels=64
        )
        
        # 6. 边缘感知模块
        self.edge_aware = EdgeAwareAttention(64)
        
        # 7. 频率分解模块
        self.freq_decomp = FrequencyDecompositionModule(64)
        
        # 8. 最终融合输出
        self.final_fusion = nn.SequentialCell([
            nn.Conv2d(64, 64, 3, padding=1, pad_mode='pad'),
            nn.PReLU(64),
            nn.Conv2d(64, 3, 3, padding=1, pad_mode='pad'),
            nn.Tanh()
        ])
        
        # 辅助操作
        self.concat = ops.Concat(axis=1)
        self.ones = ops.Ones()
        self.resize_256 = ops.ResizeBilinearV2()
        self.resize_512 = ops.ResizeBilinearV2()
        
    def construct(self, img, mask):
        """
        主网络前向传播
        
        Args:
            img: 输入图像 (B, 3, H, W), 已破损区域被mask
            mask: 破损区域掩码 (B, 1, H, W), 1表示破损
            
        Returns:
            coarse_out: 粗修复结果
            refine_out: 细修复结果  
            sr_out: 超分辨率结果
            final_out: 最终融合输出
            correspondence: 注意力分数 (用于CRA)
        """
        x = img.astype(mindspore.float32)
        shape = x.shape
        
        # 扩展mask到batch
        mask_batch = self.ones((shape[0], 1, shape[2], shape[3]), mindspore.float32)
        mask_batch = mask_batch * mask
        
        # ====== 阶段1: 粗修复 ======
        # 下采样到256x256进行粗修复
        coarse_input = self.concat([x, mask_batch])
        coarse_input = self.resize_256(coarse_input, (256, 256))
        coarse_out = self.coarse_net(coarse_input)
        
        # 上采样回512x512
        coarse_out = self.resize_512(coarse_out, (512, 512))
        
        # 用粗修复结果填充破损区域
        x_coarse = coarse_out * mask_batch + x * (1. - mask_batch)
        
        # ====== 阶段2: 细修复 ======
        refine_input = self.concat([x_coarse, mask_batch])
        refine_out, correspondence = self.refine_net(refine_input, mask)
        
        # 用细修复结果填充破损区域
        x_refine = refine_out * mask_batch + x * (1. - mask_batch)
        
        # ====== 阶段3: 超分辨率重建 ======
        sr_out, sr_intermediates = self.sr_branch(x_refine, return_intermediates=True)
        
        # ====== 阶段4: 跨模态融合 ======
        # 将修复结果和超分结果进行融合
        # 先将修复结果上采样到与超分结果相同尺寸
        refine_upsampled = self.resize_512(x_refine, (sr_out.shape[2], sr_out.shape[3]))
        
        # 跨模态注意力融合
        fused_feat = self.cross_modal_fusion(refine_upsampled, sr_out)
        
        # 边缘增强
        fused_feat = self.edge_aware(fused_feat)
        
        # 频率分解增强
        fused_feat = self.freq_decomp(fused_feat)
        
        # 最终输出
        final_out = self.final_fusion(fused_feat)
        
        return coarse_out, refine_out, sr_out, final_out, correspondence
    
    def infer_8k(self, img, mask, tile_size=512, overlap=64):
        """
        8K图像推理 (分块处理)
        
        Args:
            img: 输入图像
            mask: 破损区域掩码
            tile_size: 分块大小
            overlap: 重叠区域大小
            
        Returns:
            output: 8K修复+超分结果
        """
        # 分块推理逻辑
        h, w = img.shape[2], img.shape[3]
        
        # 计算需要的块数
        step = tile_size - overlap
        num_h = math.ceil((h - overlap) / step)
        num_w = math.ceil((w - overlap) / step)
        
        # 输出尺寸
        out_h = h * self.config.model.srgan_upscale_factor
        out_w = w * self.config.model.srgan_upscale_factor
        
        # 初始化输出和权重
        output = ops.Zeros()((1, 3, out_h, out_w), mindspore.float32)
        weight = ops.Zeros()((1, 1, out_h, out_w), mindspore.float32)
        
        # 分块处理
        for i in range(num_h):
            for j in range(num_w):
                # 计算当前块的位置
                y1 = i * step
                x1 = j * step
                y2 = min(y1 + tile_size, h)
                x2 = min(x1 + tile_size, w)
                
                # 提取块
                tile_img = img[:, :, y1:y2, x1:x2]
                tile_mask = mask[:, :, y1:y2, x1:x2]
                
                # 处理块
                _, _, _, tile_out, _ = self.construct(tile_img, tile_mask)
                
                # 计算输出位置
                out_y1 = y1 * self.config.model.srgan_upscale_factor
                out_x1 = x1 * self.config.model.srgan_upscale_factor
                out_y2 = y2 * self.config.model.srgan_upscale_factor
                out_x2 = x2 * self.config.model.srgan_upscale_factor
                
                # 融合到输出 (简化的加权融合)
                output[:, :, out_y1:out_y2, out_x1:out_x2] += tile_out
                weight[:, :, out_y1:out_y2, out_x1:out_x2] += 1.0
        
        # 归一化
        output = output / (weight + 1e-8)
        
        return output


class ContextualResidualAggregation(nn.Cell):
    """
    上下文残差聚合模块 (Contextual Residual Aggregation)
    
    创新点: 
    - 从高分辨率参考图像中聚合上下文残差
    - 通过注意力机制将背景区域的高频信息传递到破损区域
    """
    
    def __init__(self, channels):
        super(ContextualResidualAggregation, self).__init__()
        
        # 残差提取
        self.residual_extract = nn.SequentialCell([
            nn.Conv2d(channels, channels, 3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, pad_mode='pad')
        ])
        
        # 残差增强
        self.residual_enhance = nn.SequentialCell([
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        ])
        
        # 残差融合
        self.residual_fusion = nn.Conv2d(channels * 2, channels, 1)
        
        self.concat = ops.Concat(axis=1)
        
    def construct(self, low_res, high_res, correspondence):
        """
        上下文残差聚合前向传播
        
        Args:
            low_res: 低分辨率修复结果
            high_res: 高分辨率参考图像
            correspondence: 注意力分数
        """
        # 计算残差
        residual = high_res - low_res
        
        # 提取和增强残差
        residual_feat = self.residual_extract(residual)
        residual_weight = self.residual_enhance(residual_feat)
        enhanced_residual = residual_feat * residual_weight
        
        # 融合
        fusion = self.concat([low_res, enhanced_residual])
        output = self.residual_fusion(fusion)
        
        return output + low_res


def get_cra_srgan_generator(config):
    """
    获取CRA-SRGAN生成器
    
    Args:
        config: 模型配置
        
    Returns:
        CRASRGANGenerator实例
    """
    return CRASRGANGenerator(config)

