# Copyright 2024
# 模型导出脚本
"""
模型导出工具

支持格式:
- MINDIR: MindSpore原生格式,用于昇腾310推理
- ONNX: 通用格式,可用于多种推理引擎
- AIR: 昇腾离线推理格式
"""

import os
import sys
import argparse
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import get_default_config
from src.models.fusion_generator import CRASRGANGenerator


def export_model(args):
    """导出模型"""
    
    # 设置环境
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device)
    if args.device in ['GPU', 'Ascend']:
        context.set_context(device_id=args.device_id)
    
    print("=" * 60)
    print("模型导出")
    print("=" * 60)
    print(f"检查点: {args.checkpoint}")
    print(f"输出文件: {args.output}")
    print(f"输出格式: {args.file_format}")
    print(f"输入尺寸: {args.input_size}")
    print("=" * 60)
    
    # 创建模型
    print("\n加载模型...")
    config = get_default_config()
    model = CRASRGANGenerator(config)
    
    # 加载权重
    param_dict = load_checkpoint(args.checkpoint)
    load_param_into_net(model, param_dict)
    model.set_train(False)
    
    # 创建输入张量
    batch_size = args.batch_size
    input_size = args.input_size
    
    # 模型输入: (image, mask)
    image_input = Tensor(np.zeros([batch_size, 3, input_size, input_size], dtype=np.float32))
    mask_input = Tensor(np.zeros([batch_size, 1, input_size, input_size], dtype=np.float32))
    
    # 导出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 导出文件名 (不带扩展名)
    output_file = os.path.splitext(args.output)[0]
    
    print(f"\n导出模型到: {args.output}")
    
    if args.file_format == 'MINDIR':
        export(model, image_input, mask_input, file_name=output_file, file_format='MINDIR')
        print(f"MINDIR模型已导出: {output_file}.mindir")
        
    elif args.file_format == 'ONNX':
        export(model, image_input, mask_input, file_name=output_file, file_format='ONNX')
        print(f"ONNX模型已导出: {output_file}.onnx")
        
    elif args.file_format == 'AIR':
        export(model, image_input, mask_input, file_name=output_file, file_format='AIR')
        print(f"AIR模型已导出: {output_file}.air")
    
    print("\n导出完成!")
    
    # 打印模型信息
    print("\n模型信息:")
    print(f"  输入shape: image={list(image_input.shape)}, mask={list(mask_input.shape)}")
    print(f"  参数数量: {sum(p.size for p in model.get_parameters())}")


class InferenceWrapper(nn.Cell):
    """推理封装类 (简化输入输出)"""
    
    def __init__(self, generator):
        super(InferenceWrapper, self).__init__()
        self.generator = generator
    
    def construct(self, x, mask):
        """
        Args:
            x: 破损图像 (已与mask相乘)
            mask: 破损掩码
        Returns:
            修复结果
        """
        _, refine_out, _, _, _ = self.generator(x, mask)
        return refine_out


def export_inference_model(args):
    """导出推理优化模型"""
    
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device)
    
    print("=" * 60)
    print("导出推理优化模型")
    print("=" * 60)
    
    # 创建模型
    config = get_default_config()
    generator = CRASRGANGenerator(config)
    
    param_dict = load_checkpoint(args.checkpoint)
    load_param_into_net(generator, param_dict)
    
    # 封装为推理模型
    model = InferenceWrapper(generator)
    model.set_train(False)
    
    # 输入
    batch_size = args.batch_size
    input_size = args.input_size
    x_input = Tensor(np.zeros([batch_size, 3, input_size, input_size], dtype=np.float32))
    mask_input = Tensor(np.zeros([batch_size, 1, input_size, input_size], dtype=np.float32))
    
    output_file = os.path.splitext(args.output)[0] + '_inference'
    
    export(model, x_input, mask_input, file_name=output_file, file_format=args.file_format)
    
    print(f"推理模型已导出: {output_file}.{args.file_format.lower()}")


def main():
    parser = argparse.ArgumentParser(description='模型导出工具')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出文件路径')
    parser.add_argument('--file_format', type=str, default='MINDIR',
                       choices=['MINDIR', 'ONNX', 'AIR'],
                       help='导出格式')
    parser.add_argument('--input_size', type=int, default=512,
                       help='输入图像尺寸')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批大小')
    parser.add_argument('--device', type=str, default='Ascend',
                       choices=['GPU', 'Ascend', 'CPU'],
                       help='设备类型')
    parser.add_argument('--device_id', type=int, default=0,
                       help='设备ID')
    parser.add_argument('--inference_only', action='store_true',
                       help='仅导出推理模型')
    
    args = parser.parse_args()
    
    if args.inference_only:
        export_inference_model(args)
    else:
        export_model(args)


if __name__ == '__main__':
    main()

