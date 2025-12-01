# CRA-SRGAN Fusion Loss Functions
"""
混合损失函数模块

包含:
1. 重建损失 (L1, L2)
2. 感知损失 (VGG Perceptual Loss)
3. 风格损失 (Style Loss)
4. 对抗损失 (GAN Loss)
5. 边缘损失 (Edge Loss)
6. 频率损失 (Frequency Loss)
7. 修复专用损失 (Inpainting Loss)
"""

from .hybrid_loss import (
    HybridLoss,
    InpaintingLoss,
    SuperResolutionLoss,
    PerceptualLoss,
    StyleLoss,
    EdgeLoss,
    FrequencyLoss,
    AdversarialLoss,
    GeneratorLoss,
    DiscriminatorLoss
)

__all__ = [
    'HybridLoss',
    'InpaintingLoss',
    'SuperResolutionLoss',
    'PerceptualLoss',
    'StyleLoss',
    'EdgeLoss',
    'FrequencyLoss',
    'AdversarialLoss',
    'GeneratorLoss',
    'DiscriminatorLoss'
]

