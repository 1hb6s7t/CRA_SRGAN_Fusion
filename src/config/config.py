# Copyright 2024
# CRA-SRGAN Fusion Model Configuration
# 图像修复与超高清化一体化模型配置
"""
Configuration file for CRA-SRGAN Fusion Model
支持破损修复与8K超分辨率重建的端到端模型配置
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import argparse


@dataclass
class ModelConfig:
    """模型架构配置"""
    # CRA修复分支配置
    cra_input_size: int = 512                    # CRA输入分辨率
    cra_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    cra_num_res_blocks: int = 8                  # 残差块数量
    use_gated_conv: bool = True                  # 使用门控卷积
    attention_type: str = "SOFT"                 # 注意力类型: SOFT, HARD, MULTI_SCALE
    
    # SRGAN超分分支配置
    srgan_upscale_factor: int = 16               # 上采样倍数 (512->8K需要16倍)
    srgan_num_res_blocks: int = 16               # SRGAN残差块数量
    srgan_channels: int = 64                     # SRGAN基础通道数
    
    # 融合模块配置
    fusion_type: str = "attention"               # 融合方式: concat, attention, cross_attention
    fusion_channels: int = 128                   # 融合层通道数
    use_progressive_upsampling: bool = True      # 渐进式上采样
    progressive_stages: int = 4                  # 渐进式阶段数 (2x -> 4x -> 8x -> 16x)
    
    # 创新模块配置
    use_edge_aware_module: bool = True           # 边缘感知增强
    use_multi_scale_attention: bool = True       # 多尺度注意力
    use_frequency_decomposition: bool = True     # 频率分解增强
    use_texture_enhancement: bool = True         # 纹理增强模块


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    batch_size: int = 4
    learning_rate: float = 1e-4
    lr_decay_factor: float = 0.5
    lr_decay_epochs: int = 100
    total_epochs: int = 500
    
    # 分阶段训练配置
    stage1_epochs: int = 100                     # 阶段1: CRA修复预训练
    stage2_epochs: int = 100                     # 阶段2: SRGAN超分预训练
    stage3_epochs: int = 300                     # 阶段3: 联合微调
    
    # 损失权重配置
    l1_weight: float = 1.0                       # L1重建损失权重
    perceptual_weight: float = 0.1               # 感知损失权重
    style_weight: float = 0.05                   # 风格损失权重
    adversarial_weight: float = 0.001            # 对抗损失权重
    edge_weight: float = 0.1                     # 边缘一致性损失权重
    frequency_weight: float = 0.05               # 频率域损失权重
    
    # CRA特定损失权重
    in_hole_weight: float = 1.2                  # 修复区域损失权重
    context_weight: float = 1.2                  # 上下文区域损失权重
    coarse_weight: float = 1.2                   # 粗网络损失权重
    wgan_gp_lambda: float = 10.0                 # WGAN-GP损失权重
    
    # 优化器配置
    optimizer: str = "Adam"
    beta1: float = 0.5
    beta2: float = 0.9
    weight_decay: float = 0.0
    
    # 数据增强配置
    random_crop: bool = True
    random_flip: bool = True
    random_rotation: bool = True
    color_jitter: bool = True
    
    # 分布式训练配置
    use_distributed: bool = False
    device_num: int = 1
    
    # 混合精度训练
    use_mixed_precision: bool = True
    loss_scale: float = 1024.0


@dataclass
class InferenceConfig:
    """推理配置"""
    # 8K推理优化
    use_tile_inference: bool = True              # 分块推理
    tile_size: int = 512                         # 分块大小
    tile_overlap: int = 64                       # 分块重叠区域
    
    # 模型优化
    use_quantization: bool = False               # 量化推理
    quantization_bits: int = 8                   # 量化位数
    use_graph_optimization: bool = True          # 图优化
    
    # 内存优化
    max_memory_gb: float = 12.0                  # 最大显存限制
    enable_memory_optimization: bool = True      # 启用内存优化
    
    # 输出配置
    output_format: str = "png"                   # 输出格式
    output_quality: int = 100                    # 输出质量


@dataclass
class DatasetConfig:
    """数据集配置"""
    # 训练数据路径
    train_image_dir: str = "../datasets/train/images"
    train_mask_dir: str = "../datasets/train/masks"
    
    # 验证数据路径
    val_image_dir: str = "../datasets/val/images"
    val_mask_dir: str = "../datasets/val/masks"
    
    # 测试数据路径
    test_image_dir: str = "../datasets/test/images"
    test_mask_dir: str = "../datasets/test/masks"
    
    # 数据处理配置
    hr_size: Tuple[int, int] = (2048, 2048)      # 高分辨率目标尺寸
    lr_size: Tuple[int, int] = (512, 512)        # 低分辨率输入尺寸
    mask_ratio_range: Tuple[float, float] = (0.1, 0.5)  # 破损区域比例范围
    
    # 数据增强
    augmentation_prob: float = 0.5


@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str = "CRA_SRGAN_Fusion_v1"
    save_dir: str = "../checkpoints"
    log_dir: str = "../logs"
    
    # 保存配置
    save_frequency: int = 10                     # 每N个epoch保存一次
    save_best_only: bool = True
    
    # 日志配置
    log_frequency: int = 100                     # 每N步记录日志
    visualize_frequency: int = 500               # 每N步可视化
    
    # 评估配置
    eval_frequency: int = 5                      # 每N个epoch评估
    eval_metrics: List[str] = field(default_factory=lambda: [
        "PSNR", "SSIM", "LPIPS", "FID", "NIQE", "PI"
    ])


class CRASRGANConfig:
    """CRA-SRGAN融合模型总配置类"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.dataset = DatasetConfig()
        self.experiment = ExperimentConfig()
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'CRASRGANConfig':
        """从命令行参数创建配置"""
        config = cls()
        
        # 更新模型配置
        if hasattr(args, 'upscale_factor'):
            config.model.srgan_upscale_factor = args.upscale_factor
        if hasattr(args, 'attention_type'):
            config.model.attention_type = args.attention_type
            
        # 更新训练配置
        if hasattr(args, 'batch_size'):
            config.training.batch_size = args.batch_size
        if hasattr(args, 'learning_rate'):
            config.training.learning_rate = args.learning_rate
        if hasattr(args, 'epochs'):
            config.training.total_epochs = args.epochs
            
        # 更新数据集配置
        if hasattr(args, 'train_image_dir'):
            config.dataset.train_image_dir = args.train_image_dir
            
        return config
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'dataset': self.dataset.__dict__,
            'experiment': self.experiment.__dict__
        }


def get_default_config() -> CRASRGANConfig:
    """获取默认配置"""
    return CRASRGANConfig()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CRA-SRGAN Fusion Model')
    
    # 模型参数
    parser.add_argument('--upscale_factor', type=int, default=16,
                       help='Super resolution upscale factor')
    parser.add_argument('--attention_type', type=str, default='SOFT',
                       choices=['SOFT', 'HARD', 'MULTI_SCALE'])
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--device_target', type=str, default='GPU',
                       choices=['GPU', 'Ascend', 'CPU'])
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--run_distribute', type=bool, default=False)
    
    # 数据参数
    parser.add_argument('--train_image_dir', type=str, default='../datasets/train/images')
    parser.add_argument('--train_mask_dir', type=str, default='../datasets/train/masks')
    parser.add_argument('--val_image_dir', type=str, default='../datasets/val/images')
    parser.add_argument('--val_mask_dir', type=str, default='../datasets/val/masks')
    
    # 实验参数
    parser.add_argument('--experiment_name', type=str, default='CRA_SRGAN_Fusion_v1')
    parser.add_argument('--save_dir', type=str, default='../checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # 推理参数
    parser.add_argument('--use_tile_inference', type=bool, default=True)
    parser.add_argument('--tile_size', type=int, default=512)
    
    return parser.parse_args()


if __name__ == '__main__':
    # 测试配置
    config = get_default_config()
    print("Model Config:", config.model)
    print("Training Config:", config.training)

