# CRA-SRGAN Fusion Model
"""
图像修复与超高清化一体化模型

核心创新点:
1. 上下文残差聚合增强的多尺度注意力机制 (CRA-Enhanced Multi-Scale Attention)
2. 渐进式修复-超分联合学习框架 (Progressive Joint Learning Framework)
3. 边缘感知的高频细节保真模块 (Edge-Aware High-Frequency Fidelity Module)
"""

from .fusion_generator import CRASRGANGenerator
from .fusion_discriminator import MultiScaleDiscriminator
from .attention_modules import (
    MultiScaleContextualAttention,
    CrossModalFusionAttention,
    EdgeAwareAttention
)
from .network_modules import (
    GatedConv2d,
    TransposeGatedConv2d,
    ResidualBlock,
    SubpixelConvolutionLayer
)

__all__ = [
    'CRASRGANGenerator',
    'MultiScaleDiscriminator',
    'MultiScaleContextualAttention',
    'CrossModalFusionAttention',
    'EdgeAwareAttention',
    'GatedConv2d',
    'TransposeGatedConv2d',
    'ResidualBlock',
    'SubpixelConvolutionLayer'
]

