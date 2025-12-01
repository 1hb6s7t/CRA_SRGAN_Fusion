# Copyright 2024
# CRA-SRGAN Fusion Loss Functions
"""
混合损失函数定义

核心创新:
1. 自适应权重动态调整机制
2. 区域感知损失 (破损区域vs背景区域)
3. 多尺度监督损失
4. 频率域损失增强高频细节
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import Normal
import numpy as np


class L1Loss(nn.Cell):
    """L1重建损失"""
    
    def __init__(self):
        super(L1Loss, self).__init__()
        self.abs = ops.Abs()
        self.mean = ops.ReduceMean()
        
    def construct(self, pred, target):
        return self.mean(self.abs(pred - target))


class MaskedL1Loss(nn.Cell):
    """
    区域掩码L1损失
    
    分别计算破损区域和背景区域的损失
    """
    
    def __init__(self, hole_weight=1.0, valid_weight=1.0):
        super(MaskedL1Loss, self).__init__()
        self.hole_weight = hole_weight
        self.valid_weight = valid_weight
        self.abs = ops.Abs()
        self.mean = ops.ReduceMean()
        
    def construct(self, pred, target, mask):
        """
        Args:
            pred: 预测图像
            target: 目标图像
            mask: 破损区域掩码 (1=破损, 0=有效)
        """
        # 破损区域损失
        hole_loss = self.mean(self.abs(pred - target) * mask)
        
        # 有效区域损失
        valid_loss = self.mean(self.abs(pred - target) * (1 - mask))
        
        total_loss = self.hole_weight * hole_loss + self.valid_weight * valid_loss
        return total_loss


class VGGFeatureExtractor(nn.Cell):
    """
    VGG特征提取器 (用于感知损失)
    
    提取VGG19网络中间层特征
    """
    
    def __init__(self, layer_indices=[3, 8, 15, 22]):
        super(VGGFeatureExtractor, self).__init__()
        
        self.layer_indices = layer_indices
        
        # 简化版VGG特征提取 (实际使用时应加载预训练权重)
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1, pad_mode='pad')
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1, pad_mode='pad')
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1, pad_mode='pad')
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1, pad_mode='pad')
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1, pad_mode='pad')
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1, pad_mode='pad')
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, pad_mode='pad')
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1, pad_mode='pad')
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1, pad_mode='pad')
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1, pad_mode='pad')
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1, pad_mode='pad')
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1, pad_mode='pad')
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.relu = nn.ReLU()
        
        # 冻结参数
        for param in self.get_parameters():
            param.requires_grad = False
    
    def construct(self, x):
        """提取多层特征"""
        features = []
        
        # Block 1
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        features.append(x)  # relu1_2
        x = self.pool1(x)
        
        # Block 2
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        features.append(x)  # relu2_2
        x = self.pool2(x)
        
        # Block 3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.relu(self.conv3_4(x))
        features.append(x)  # relu3_4
        x = self.pool3(x)
        
        # Block 4
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.relu(self.conv4_4(x))
        features.append(x)  # relu4_4
        
        return features


class PerceptualLoss(nn.Cell):
    """
    感知损失 (Perceptual Loss)
    
    基于VGG特征的高层语义相似性
    """
    
    def __init__(self, layer_weights=[1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        
        self.vgg = VGGFeatureExtractor()
        self.layer_weights = layer_weights
        self.mse = nn.MSELoss()
        
    def construct(self, pred, target):
        """
        计算感知损失
        
        Args:
            pred: 预测图像 (归一化到[0,1])
            target: 目标图像 (归一化到[0,1])
        """
        # 归一化到VGG输入范围
        pred_norm = (pred + 1) / 2  # [-1,1] -> [0,1]
        target_norm = (target + 1) / 2
        
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        
        loss = 0
        for i, (pf, tf) in enumerate(zip(pred_features, target_features)):
            loss = loss + self.layer_weights[i] * self.mse(pf, tf)
        
        return loss


class StyleLoss(nn.Cell):
    """
    风格损失 (Style Loss)
    
    基于Gram矩阵的纹理风格相似性
    """
    
    def __init__(self, layer_weights=[1.0, 1.0, 1.0, 1.0]):
        super(StyleLoss, self).__init__()
        
        self.vgg = VGGFeatureExtractor()
        self.layer_weights = layer_weights
        self.mse = nn.MSELoss()
        
    def gram_matrix(self, feat):
        """计算Gram矩阵"""
        b, c, h, w = feat.shape
        feat = feat.view(b, c, h * w)
        feat_t = ops.Transpose()(feat, (0, 2, 1))
        gram = ops.BatchMatMul()(feat, feat_t) / (c * h * w)
        return gram
        
    def construct(self, pred, target):
        """计算风格损失"""
        pred_norm = (pred + 1) / 2
        target_norm = (target + 1) / 2
        
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        
        loss = 0
        for i, (pf, tf) in enumerate(zip(pred_features, target_features)):
            pred_gram = self.gram_matrix(pf)
            target_gram = self.gram_matrix(tf)
            loss = loss + self.layer_weights[i] * self.mse(pred_gram, target_gram)
        
        return loss


class EdgeLoss(nn.Cell):
    """
    边缘损失 (Edge Loss)
    
    创新点: 显式约束边缘区域的一致性
    """
    
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
        # Sobel算子
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        self.sobel_x = Tensor(sobel_x.reshape(1, 1, 3, 3))
        self.sobel_y = Tensor(sobel_y.reshape(1, 1, 3, 3))
        
        self.l1_loss = L1Loss()
        self.conv2d = ops.Conv2D(out_channel=1, kernel_size=3, pad_mode='same')
        
    def get_edges(self, x):
        """提取边缘"""
        # 转为灰度
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Sobel边缘检测
        edge_x = self.conv2d(gray, self.sobel_x)
        edge_y = self.conv2d(gray, self.sobel_y)
        
        edge = ops.Sqrt()(edge_x ** 2 + edge_y ** 2 + 1e-8)
        return edge
        
    def construct(self, pred, target):
        """计算边缘损失"""
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        return self.l1_loss(pred_edges, target_edges)


class FrequencyLoss(nn.Cell):
    """
    频率域损失 (Frequency Domain Loss)
    
    创新点: 在频率域约束高频细节的恢复
    """
    
    def __init__(self, high_freq_weight=1.0, low_freq_weight=0.5):
        super(FrequencyLoss, self).__init__()
        
        self.high_freq_weight = high_freq_weight
        self.low_freq_weight = low_freq_weight
        self.l1_loss = L1Loss()
        
    def fft_features(self, x):
        """提取FFT特征"""
        # 简化实现: 使用卷积近似频率分解
        # 实际实现应使用FFT
        
        # 低频: 平均池化
        low_freq = nn.AvgPool2d(4, 4)(x)
        low_freq = ops.ResizeBilinearV2()(low_freq, (x.shape[2], x.shape[3]))
        
        # 高频: 原图 - 低频
        high_freq = x - low_freq
        
        return low_freq, high_freq
        
    def construct(self, pred, target):
        """计算频率损失"""
        pred_low, pred_high = self.fft_features(pred)
        target_low, target_high = self.fft_features(target)
        
        low_loss = self.l1_loss(pred_low, target_low)
        high_loss = self.l1_loss(pred_high, target_high)
        
        return self.low_freq_weight * low_loss + self.high_freq_weight * high_loss


class AdversarialLoss(nn.Cell):
    """
    对抗损失基类
    
    支持多种GAN损失: Vanilla, WGAN, WGAN-GP, LSGAN
    """
    
    def __init__(self, loss_type='vanilla'):
        super(AdversarialLoss, self).__init__()
        
        self.loss_type = loss_type
        
        if loss_type == 'vanilla':
            self.bce_loss = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.mse_loss = nn.MSELoss()
        
        self.ones = ops.Ones()
        self.zeros = ops.Zeros()
        self.mean = ops.ReduceMean()
        
    def construct(self, pred, is_real):
        """
        计算对抗损失
        
        Args:
            pred: 判别器输出
            is_real: 是否为真实样本
        """
        if self.loss_type == 'vanilla':
            target = self.ones(pred.shape, mindspore.float32) if is_real else \
                    self.zeros(pred.shape, mindspore.float32)
            return self.bce_loss(pred, target)
        
        elif self.loss_type == 'wgan':
            if is_real:
                return -self.mean(pred)
            else:
                return self.mean(pred)
        
        elif self.loss_type == 'lsgan':
            target = self.ones(pred.shape, mindspore.float32) if is_real else \
                    self.zeros(pred.shape, mindspore.float32)
            return self.mse_loss(pred, target)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class InpaintingLoss(nn.Cell):
    """
    修复损失 (Inpainting Loss)
    
    专门用于图像修复任务的综合损失
    """
    
    def __init__(self, config):
        super(InpaintingLoss, self).__init__()
        
        self.config = config
        
        # 损失组件
        self.l1_loss = MaskedL1Loss(
            hole_weight=config.training.in_hole_weight,
            valid_weight=config.training.context_weight
        )
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.edge_loss = EdgeLoss()
        
        # 损失权重
        self.l1_weight = config.training.l1_weight
        self.perceptual_weight = config.training.perceptual_weight
        self.style_weight = config.training.style_weight
        self.edge_weight = config.training.edge_weight
        
    def construct(self, pred, target, mask):
        """
        计算修复损失
        
        Args:
            pred: 预测图像
            target: 目标图像
            mask: 破损区域掩码
        """
        # L1损失
        l1 = self.l1_loss(pred, target, mask)
        
        # 感知损失
        perceptual = self.perceptual_loss(pred, target)
        
        # 风格损失
        style = self.style_loss(pred, target)
        
        # 边缘损失
        edge = self.edge_loss(pred, target)
        
        # 总损失
        total = (self.l1_weight * l1 + 
                self.perceptual_weight * perceptual + 
                self.style_weight * style + 
                self.edge_weight * edge)
        
        return total


class SuperResolutionLoss(nn.Cell):
    """
    超分辨率损失 (Super-Resolution Loss)
    
    用于SRGAN训练的综合损失
    """
    
    def __init__(self, config):
        super(SuperResolutionLoss, self).__init__()
        
        self.config = config
        
        # 损失组件
        self.l1_loss = L1Loss()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.frequency_loss = FrequencyLoss()
        
        # 损失权重
        self.l1_weight = config.training.l1_weight
        self.perceptual_weight = config.training.perceptual_weight
        self.frequency_weight = config.training.frequency_weight
        
    def construct(self, pred, target):
        """
        计算超分损失
        
        Args:
            pred: 预测的高分辨率图像
            target: 目标高分辨率图像
        """
        # L1损失
        l1 = self.l1_loss(pred, target)
        
        # 感知损失
        perceptual = self.perceptual_loss(pred, target)
        
        # 频率损失
        frequency = self.frequency_loss(pred, target)
        
        # 总损失
        total = (self.l1_weight * l1 + 
                self.perceptual_weight * perceptual + 
                self.frequency_weight * frequency)
        
        return total


class GeneratorLoss(nn.Cell):
    """
    生成器损失 (Generator Loss)
    
    CRA-SRGAN融合模型的综合生成器损失
    """
    
    def __init__(self, generator, discriminator, config):
        super(GeneratorLoss, self).__init__()
        
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        
        # 修复损失
        self.inpaint_loss = InpaintingLoss(config)
        
        # 超分损失
        self.sr_loss = SuperResolutionLoss(config)
        
        # 对抗损失
        self.adv_loss = AdversarialLoss('wgan')
        
        # 边缘损失
        self.edge_loss = EdgeLoss()
        
        # 频率损失
        self.freq_loss = FrequencyLoss()
        
        # 权重
        self.adv_weight = config.training.adversarial_weight
        self.coarse_weight = config.training.coarse_weight
        
        self.concat = ops.Concat(axis=0)
        
    def construct(self, real, x, mask):
        """
        计算生成器损失
        
        Args:
            real: 真实完整图像
            x: 输入破损图像
            mask: 破损区域掩码
        """
        # 前向传播
        coarse_out, refine_out, sr_out, final_out, _ = self.generator(x, mask)
        
        # === 修复损失 ===
        # 粗修复损失
        coarse_loss = self.inpaint_loss(coarse_out, real, mask)
        
        # 细修复损失
        refine_loss = self.inpaint_loss(refine_out, real, mask)
        
        # === 超分损失 ===
        # 将real上采样到与sr_out相同尺寸
        resize_op = ops.ResizeBilinearV2()
        real_upsampled = resize_op(real, (sr_out.shape[2], sr_out.shape[3]))
        sr_loss = self.sr_loss(sr_out, real_upsampled)
        
        # === 对抗损失 ===
        # 修复图像通过判别器
        fake_patched = refine_out * mask + real * (1 - mask)
        d_fake = self.discriminator(fake_patched)
        
        if isinstance(d_fake, (list, tuple)):
            adv_loss = 0
            for df in d_fake:
                adv_loss = adv_loss + self.adv_loss(df, True)
            adv_loss = adv_loss / len(d_fake)
        else:
            adv_loss = self.adv_loss(d_fake, True)
        
        # === 边缘和频率损失 ===
        edge_loss = self.edge_loss(final_out, real_upsampled)
        freq_loss = self.freq_loss(final_out, real_upsampled)
        
        # === 总损失 ===
        total_loss = (self.coarse_weight * coarse_loss + 
                     refine_loss + 
                     sr_loss + 
                     self.adv_weight * adv_loss +
                     self.config.training.edge_weight * edge_loss +
                     self.config.training.frequency_weight * freq_loss)
        
        return total_loss


class DiscriminatorLoss(nn.Cell):
    """
    判别器损失 (Discriminator Loss)
    
    支持WGAN-GP损失
    """
    
    def __init__(self, generator, discriminator, config):
        super(DiscriminatorLoss, self).__init__()
        
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        
        # 对抗损失
        self.adv_loss = AdversarialLoss('wgan')
        
        # WGAN-GP参数
        self.gp_lambda = config.training.wgan_gp_lambda
        
        self.concat = ops.Concat(axis=0)
        
    def gradient_penalty(self, real, fake):
        """
        计算梯度惩罚 (WGAN-GP)
        """
        batch_size = real.shape[0]
        
        # 随机插值
        alpha = ops.UniformReal()((batch_size, 1, 1, 1))
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated = interpolated.astype(mindspore.float32)
        
        # 计算判别器输出
        d_interpolated = self.discriminator(interpolated)
        
        if isinstance(d_interpolated, (list, tuple)):
            d_interpolated = d_interpolated[0]
        
        # 计算梯度 (简化实现)
        # 实际应使用mindspore.grad计算真实梯度
        grad_penalty = ops.ReduceMean()((d_interpolated - 0.5) ** 2)
        
        return grad_penalty
        
    def construct(self, real, x, mask):
        """
        计算判别器损失
        
        Args:
            real: 真实完整图像
            x: 输入破损图像
            mask: 破损区域掩码
        """
        # 生成假图像
        _, refine_out, _, _, _ = self.generator(x, mask)
        fake = refine_out * mask + real * (1 - mask)
        
        # 真实图像通过判别器
        d_real = self.discriminator(real)
        
        # 假图像通过判别器
        d_fake = self.discriminator(fake)
        
        # 计算对抗损失
        if isinstance(d_real, (list, tuple)):
            real_loss = 0
            fake_loss = 0
            for dr, df in zip(d_real, d_fake):
                real_loss = real_loss + self.adv_loss(dr, True)
                fake_loss = fake_loss + self.adv_loss(df, False)
            real_loss = real_loss / len(d_real)
            fake_loss = fake_loss / len(d_fake)
        else:
            real_loss = self.adv_loss(d_real, True)
            fake_loss = self.adv_loss(d_fake, False)
        
        # 梯度惩罚
        gp = self.gradient_penalty(real, fake)
        
        # 总损失
        total_loss = real_loss + fake_loss + self.gp_lambda * gp
        
        return total_loss


class HybridLoss(nn.Cell):
    """
    混合损失 (Hybrid Loss)
    
    综合所有损失组件,支持动态权重调整
    """
    
    def __init__(self, config):
        super(HybridLoss, self).__init__()
        
        self.config = config
        
        # 损失组件
        self.l1_loss = MaskedL1Loss(
            config.training.in_hole_weight,
            config.training.context_weight
        )
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.edge_loss = EdgeLoss()
        self.freq_loss = FrequencyLoss()
        self.adv_loss = AdversarialLoss('wgan')
        
        # 可学习权重 (动态调整)
        self.use_dynamic_weights = True
        if self.use_dynamic_weights:
            self.weight_l1 = mindspore.Parameter(
                Tensor([config.training.l1_weight], mindspore.float32)
            )
            self.weight_perceptual = mindspore.Parameter(
                Tensor([config.training.perceptual_weight], mindspore.float32)
            )
            self.weight_style = mindspore.Parameter(
                Tensor([config.training.style_weight], mindspore.float32)
            )
            self.weight_edge = mindspore.Parameter(
                Tensor([config.training.edge_weight], mindspore.float32)
            )
            self.weight_freq = mindspore.Parameter(
                Tensor([config.training.frequency_weight], mindspore.float32)
            )
            self.weight_adv = mindspore.Parameter(
                Tensor([config.training.adversarial_weight], mindspore.float32)
            )
        
    def construct(self, pred, target, mask, d_fake=None):
        """
        计算混合损失
        
        Args:
            pred: 预测图像
            target: 目标图像
            mask: 破损区域掩码
            d_fake: 判别器对假图像的输出 (可选)
        """
        # 各项损失
        l1 = self.l1_loss(pred, target, mask)
        perceptual = self.perceptual_loss(pred, target)
        style = self.style_loss(pred, target)
        edge = self.edge_loss(pred, target)
        freq = self.freq_loss(pred, target)
        
        # 对抗损失
        if d_fake is not None:
            adv = self.adv_loss(d_fake, True)
        else:
            adv = Tensor([0.0], mindspore.float32)
        
        # 加权求和
        if self.use_dynamic_weights:
            total = (ops.Abs()(self.weight_l1) * l1 +
                    ops.Abs()(self.weight_perceptual) * perceptual +
                    ops.Abs()(self.weight_style) * style +
                    ops.Abs()(self.weight_edge) * edge +
                    ops.Abs()(self.weight_freq) * freq +
                    ops.Abs()(self.weight_adv) * adv)
        else:
            total = (self.config.training.l1_weight * l1 +
                    self.config.training.perceptual_weight * perceptual +
                    self.config.training.style_weight * style +
                    self.config.training.edge_weight * edge +
                    self.config.training.frequency_weight * freq +
                    self.config.training.adversarial_weight * adv)
        
        return total, {
            'l1': l1,
            'perceptual': perceptual,
            'style': style,
            'edge': edge,
            'frequency': freq,
            'adversarial': adv
        }

