# CRA-SRGAN 图像修复与超高清化模型
# GPU版本完整使用手册

---

## 目录

1. [硬件和环境要求](#一硬件和环境要求)
2. [环境安装配置](#二环境安装配置)
3. [数据集准备](#三数据集准备)
4. [模型训练](#四模型训练)
5. [模型推理](#五模型推理)
6. [评估与可视化](#六评估与可视化)
7. [常见问题](#七常见问题)

---

## 一、硬件和环境要求

### 1.1 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| **GPU** | NVIDIA GTX 1080 Ti (11GB显存) | NVIDIA RTX 3090/4090 (24GB显存) |
| **CPU** | Intel i7-8700 / AMD Ryzen 7 | Intel i9 / AMD Ryzen 9 |
| **内存** | 32GB | 64GB+ |
| **硬盘** | 256GB SSD | 1TB NVMe SSD |
| **CUDA** | 11.1+ | 11.6+ |
| **cuDNN** | 8.0+ | 8.4+ |

### 1.2 软件要求

| 软件 | 版本要求 |
|------|----------|
| 操作系统 | Ubuntu 18.04/20.04 或 Windows 10/11 |
| Python | 3.8 - 3.10 |
| MindSpore | 2.2.0+ (GPU版本) |
| CUDA | 11.1+ |
| cuDNN | 8.0+ |

### 1.3 检查GPU环境

```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查CUDA版本
nvcc --version

# 应看到类似输出:
# CUDA compilation tools, release 11.6
```

---

## 二、环境安装配置

### 2.1 创建虚拟环境

**方式1: Conda (推荐)**

```bash
# 创建conda环境
conda create -n cra_srgan python=3.8 -y

# 激活环境
conda activate cra_srgan

# 验证Python版本
python --version
# 输出: Python 3.8.x
```

**方式2: venv**

```bash
# 创建虚拟环境
python -m venv cra_srgan_env

# 激活环境
# Linux/Mac:
source cra_srgan_env/bin/activate
# Windows:
cra_srgan_env\Scripts\activate
```

### 2.2 安装MindSpore GPU版本

```bash
# 根据CUDA版本安装MindSpore
# CUDA 11.1:
pip install mindspore==2.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# CUDA 11.6:
pip install mindspore==2.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证安装
python -c "import mindspore; print(mindspore.__version__); print(mindspore.context.get_context('device_target'))"
```

### 2.3 安装项目依赖

```bash
# 进入项目目录
cd CRA_SRGAN_Fusion

# 安装基础依赖
pip install -r requirements.txt

# 安装额外的图像处理库
pip install opencv-python-headless albumentations

# (可选) 安装医学影像处理库
pip install pydicom nibabel SimpleITK
```

### 2.4 验证环境

```bash
# 运行验证脚本
python -c "
import mindspore as ms
from mindspore import context

# 设置GPU
context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

print('MindSpore版本:', ms.__version__)
print('设备:', context.get_context('device_target'))
print('GPU环境配置成功!')
"
```

**预期输出:**
```
MindSpore版本: 2.2.0
设备: GPU
GPU环境配置成功!
```

---

## 三、数据集准备

### 3.1 目录结构创建

```bash
# 在项目根目录执行
cd CRA_SRGAN_Fusion

# 创建数据目录
mkdir -p datasets/cultural_relics/train/images
mkdir -p datasets/cultural_relics/train/masks
mkdir -p datasets/cultural_relics/val/images
mkdir -p datasets/cultural_relics/val/masks
mkdir -p datasets/cultural_relics/test/images
mkdir -p datasets/cultural_relics/test/masks

mkdir -p datasets/medical/train/images
mkdir -p datasets/medical/train/masks
mkdir -p datasets/medical/val/images
mkdir -p datasets/medical/val/masks
mkdir -p datasets/medical/test/images
mkdir -p datasets/medical/test/masks

# 创建输出目录
mkdir -p outputs checkpoints/cultural_relics checkpoints/medical logs
```

### 3.2 数据准备方式

#### 方式A: 使用自动数据准备脚本

```bash
# 文物修复数据准备 (会自动生成破损掩码)
python scripts/prepare_data.py \
    --mode cultural_relics \
    --input_dir /path/to/your/raw_images \
    --output_dir ./datasets/cultural_relics \
    --target_size 512 \
    --masks_per_image 3 \
    --split_ratio 0.9

# 医学影像数据准备
python scripts/prepare_data.py \
    --mode medical \
    --input_dir /path/to/your/medical_images \
    --output_dir ./datasets/medical \
    --target_size 512 \
    --masks_per_image 2 \
    --split_ratio 0.9 \
    --grayscale
```

#### 方式B: 手动准备数据

**数据要求:**
- 图像格式: PNG, JPG, JPEG, BMP
- 掩码格式: PNG (灰度图, 白色=破损区域, 黑色=正常区域)
- 建议尺寸: 512x512

**目录结构示例:**
```
datasets/cultural_relics/train/
├── images/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
└── masks/
    ├── 000001.png  # 与图像同名
    ├── 000002.png
    └── ...
```

### 3.3 数据增强配置

修改 `configs/cultural_relics.yaml` 或 `configs/medical.yaml`:

```yaml
# 数据增强配置
training:
  random_crop: true      # 随机裁剪
  random_flip: true      # 随机翻转
  random_rotation: true  # 随机旋转 (文物修复启用)
  color_jitter: false    # 颜色抖动 (文物修复禁用)
```

### 3.4 验证数据集

```bash
# 检查数据集
python -c "
import os
from glob import glob

train_images = glob('./datasets/cultural_relics/train/images/*')
train_masks = glob('./datasets/cultural_relics/train/masks/*')
val_images = glob('./datasets/cultural_relics/val/images/*')

print(f'训练图像数量: {len(train_images)}')
print(f'训练掩码数量: {len(train_masks)}')
print(f'验证图像数量: {len(val_images)}')
print(f'数据准备完成!' if len(train_images) > 0 else '警告: 未找到训练数据!')
"
```

---

## 四、模型训练

### 4.1 训练配置

#### 文物修复配置 (configs/cultural_relics.yaml)

```yaml
model:
  cra_input_size: 512
  srgan_upscale_factor: 4
  use_edge_aware_module: true
  use_texture_enhancement: true

training:
  batch_size: 4              # 根据显存调整
  learning_rate: 0.0001
  total_epochs: 300
  l1_weight: 1.0
  perceptual_weight: 0.2     # 保留风格
  style_weight: 0.15         # 保留纹理
  adversarial_weight: 0.0005
  edge_weight: 0.15
```

#### 医学影像配置 (configs/medical.yaml)

```yaml
model:
  cra_input_size: 512
  srgan_upscale_factor: 2
  use_edge_aware_module: true

training:
  batch_size: 4
  learning_rate: 0.00005     # 较低学习率
  total_epochs: 200
  l1_weight: 2.0             # 高像素精度
  perceptual_weight: 0.05
  style_weight: 0.01
  adversarial_weight: 0.0001 # 最小化GAN影响
  edge_weight: 0.2
```

### 4.2 开始训练

#### 文物修复训练

```bash
# 单GPU训练
python scripts/train_cultural_relics.py \
    --image_dir ./datasets/cultural_relics/train/images \
    --mask_dir ./datasets/cultural_relics/train/masks \
    --batch_size 4 \
    --epochs 300 \
    --lr 0.0001 \
    --device GPU \
    --device_id 0 \
    --save_dir ./checkpoints/cultural_relics \
    --log_interval 50 \
    --save_interval 20
```

#### 医学影像训练

```bash
# 单GPU训练
python scripts/train_medical.py \
    --image_dir ./datasets/medical/train/images \
    --mask_dir ./datasets/medical/train/masks \
    --batch_size 4 \
    --epochs 200 \
    --lr 0.00005 \
    --device GPU \
    --device_id 0 \
    --save_dir ./checkpoints/medical \
    --log_interval 50 \
    --save_interval 20
```

### 4.3 多GPU分布式训练

```bash
# 4卡分布式训练
mpirun -n 4 python train.py \
    --train_image_dir ./datasets/cultural_relics/train/images \
    --train_mask_dir ./datasets/cultural_relics/train/masks \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --epochs 300 \
    --device_target GPU \
    --run_distribute True \
    --device_num 4
```

### 4.4 断点续训

```bash
# 从断点继续训练
python scripts/train_cultural_relics.py \
    --image_dir ./datasets/cultural_relics/train/images \
    --mask_dir ./datasets/cultural_relics/train/masks \
    --resume ./checkpoints/cultural_relics/generator_epoch100.ckpt \
    --epochs 300 \
    --device GPU
```

### 4.5 监控训练

```bash
# 新终端窗口启动TensorBoard
tensorboard --logdir=./checkpoints/cultural_relics/logs --port=6006

# 浏览器访问 http://localhost:6006
```

### 4.6 显存不足解决方案

```bash
# 方案1: 减小batch_size
--batch_size 2

# 方案2: 减小输入尺寸 (修改配置文件)
cra_input_size: 384

# 方案3: 使用梯度累积 (等效更大batch_size)
# 在代码中设置 accumulation_steps=2
```

---

## 五、模型推理

### 5.1 单张图像推理

```bash
# 基础推理
python infer.py \
    --input ./test_images/damaged.jpg \
    --mask ./test_images/mask.png \
    --output ./outputs/restored.png \
    --checkpoint ./checkpoints/cultural_relics/best_generator.ckpt \
    --device GPU

# 指定GPU
python infer.py \
    --input ./test_images/damaged.jpg \
    --mask ./test_images/mask.png \
    --output ./outputs/restored.png \
    --checkpoint ./checkpoints/cultural_relics/best_generator.ckpt \
    --device GPU \
    --device_id 0
```

### 5.2 批量推理

```bash
# 批量处理整个目录
python infer.py \
    --input ./test_images/batch/ \
    --mask ./test_images/masks/ \
    --output ./outputs/batch/ \
    --checkpoint ./checkpoints/cultural_relics/best_generator.ckpt \
    --device GPU \
    --batch_size 4
```

### 5.3 高分辨率推理 (分块处理)

```bash
# 大尺寸图像分块推理 (4K/8K)
python infer.py \
    --input ./test_images/high_res.jpg \
    --mask ./test_images/high_res_mask.png \
    --output ./outputs/high_res_restored.png \
    --checkpoint ./checkpoints/cultural_relics/best_generator.ckpt \
    --device GPU \
    --mode 8k \
    --tile_size 512 \
    --tile_overlap 64
```

### 5.4 Python API推理

```python
# inference_example.py
import os
import cv2
import numpy as np
import mindspore as ms
from mindspore import Tensor, context, load_checkpoint, load_param_into_net

# 设置GPU环境
context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=0)

# 导入模型
from src.config.config import get_default_config
from src.models.fusion_generator import CRASRGANGenerator


def load_model(checkpoint_path):
    """加载模型"""
    config = get_default_config()
    model = CRASRGANGenerator(config)
    
    param_dict = load_checkpoint(checkpoint_path)
    load_param_into_net(model, param_dict)
    model.set_train(False)
    
    return model


def preprocess(image_path, mask_path, size=512):
    """预处理"""
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]
    img = cv2.resize(img, (size, size))
    
    # 读取掩码
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (size, size))
    
    # 归一化
    img = img.astype(np.float32) / 127.5 - 1
    mask = mask.astype(np.float32) / 255.0
    
    # 转换为Tensor
    img_tensor = Tensor(img.transpose(2, 0, 1)[np.newaxis, ...], ms.float32)
    mask_tensor = Tensor(mask[np.newaxis, np.newaxis, ...], ms.float32)
    
    return img_tensor, mask_tensor, original_size


def postprocess(output, original_size):
    """后处理"""
    output = output.asnumpy()[0].transpose(1, 2, 0)
    output = ((output + 1) * 127.5).clip(0, 255).astype(np.uint8)
    output = cv2.resize(output, (original_size[1], original_size[0]))
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


def inpaint(model, image_path, mask_path, output_path):
    """执行修复"""
    img, mask, orig_size = preprocess(image_path, mask_path)
    
    # 创建破损输入
    x = img * (1 - mask)
    
    # 推理
    _, refine_out, _, _, _ = model(x, mask)
    
    # 后处理并保存
    result = postprocess(refine_out, orig_size)
    cv2.imwrite(output_path, result)
    print(f"保存到: {output_path}")


# 使用示例
if __name__ == '__main__':
    model = load_model('./checkpoints/cultural_relics/best_generator.ckpt')
    inpaint(model, 'damaged.jpg', 'mask.png', 'restored.png')
```

---

## 六、评估与可视化

### 6.1 模型评估

```bash
# 全图评估
python scripts/evaluate.py \
    --gt_dir ./datasets/cultural_relics/test/images \
    --pred_dir ./outputs/test_results \
    --output ./evaluation_results.csv

# 仅评估修复区域
python scripts/evaluate.py \
    --gt_dir ./datasets/cultural_relics/test/images \
    --pred_dir ./outputs/test_results \
    --mask_dir ./datasets/cultural_relics/test/masks \
    --mode inpainting \
    --output ./inpainting_evaluation.csv
```

### 6.2 结果可视化

```bash
# 单张对比图
python scripts/visualize.py \
    --mode single \
    --original ./test_images/original.jpg \
    --mask ./test_images/mask.png \
    --restored ./outputs/restored.png \
    --output ./comparison.png

# 批量生成对比图
python scripts/visualize.py \
    --mode batch \
    --original ./datasets/cultural_relics/test/images \
    --mask ./datasets/cultural_relics/test/masks \
    --restored ./outputs/test_results \
    --output ./visualization \
    --num_samples 20
```

---

## 七、常见问题

### Q1: CUDA out of memory

```bash
# 解决方案
1. 减小batch_size: --batch_size 2
2. 减小输入尺寸: 修改配置 cra_input_size: 384
3. 释放显存: 
   import gc
   gc.collect()
   mindspore.ms_memory_recycle()
```

### Q2: 训练Loss不下降

```bash
# 检查步骤
1. 确认数据路径正确
2. 检查掩码格式 (白色=破损)
3. 降低学习率: --lr 0.00005
4. 增加L1权重
```

### Q3: 修复结果模糊

```yaml
# 调整配置
training:
  perceptual_weight: 0.3  # 增加感知损失
  edge_weight: 0.2        # 增加边缘损失
```

### Q4: 多GPU训练报错

```bash
# 检查MPI环境
which mpirun
mpirun --version

# 确保所有GPU可见
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## 附录: 完整训练命令汇总

```bash
# ==================== 文物修复 ====================
# 数据准备
python scripts/prepare_data.py --mode cultural_relics --input_dir /path/to/images --output_dir ./datasets/cultural_relics

# 单卡训练
python scripts/train_cultural_relics.py --image_dir ./datasets/cultural_relics/train/images --mask_dir ./datasets/cultural_relics/train/masks --epochs 300 --batch_size 4 --device GPU

# 推理
python infer.py --input test.jpg --mask mask.png --output result.png --checkpoint ./checkpoints/cultural_relics/best_generator.ckpt --device GPU

# ==================== 医学影像 ====================
# 数据准备
python scripts/prepare_data.py --mode medical --input_dir /path/to/medical --output_dir ./datasets/medical --grayscale

# 单卡训练
python scripts/train_medical.py --image_dir ./datasets/medical/train/images --mask_dir ./datasets/medical/train/masks --epochs 200 --batch_size 4 --device GPU

# 推理
python infer.py --input xray.png --mask artifact_mask.png --output xray_restored.png --checkpoint ./checkpoints/medical/best_generator.ckpt --device GPU
```

