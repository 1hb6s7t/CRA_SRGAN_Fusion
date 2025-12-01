# CRA-SRGAN 图像修复与超高清化模型
# 昇腾(Ascend)版本完整使用手册

---

## 目录

1. [硬件和环境要求](#一硬件和环境要求)
2. [环境安装配置](#二环境安装配置)
3. [数据集准备](#三数据集准备)
4. [模型训练](#四模型训练)
5. [模型推理](#五模型推理)
6. [评估与可视化](#六评估与可视化)
7. [性能优化](#七性能优化)
8. [常见问题](#八常见问题)

---

## 一、硬件和环境要求

### 1.1 支持的昇腾硬件

| 硬件 | 类型 | 推荐场景 |
|------|------|----------|
| **Atlas 300I** | 推理卡 | 模型推理部署 |
| **Atlas 300T** | 训练卡 | 模型训练 |
| **Atlas 800 训练服务器** | 服务器 | 大规模分布式训练 |
| **Atlas 900 集群** | 集群 | 超大规模训练 |
| **Ascend 310** | 边缘推理 | 端侧推理 |
| **Ascend 910** | 训练芯片 | 训练和推理 |

### 1.2 软件环境要求

| 软件 | 版本要求 |
|------|----------|
| 操作系统 | EulerOS 2.8/2.9, Ubuntu 18.04/20.04, CentOS 7.6/8.2 |
| Python | 3.7 - 3.9 |
| MindSpore | 2.2.0+ (Ascend版本) |
| CANN | 6.0.0+ |
| 驱动 | 对应CANN版本的驱动 |

### 1.3 检查昇腾环境

```bash
# 检查NPU设备
npu-smi info

# 预期输出示例:
# +------------------------------------------------------------------------------------+
# | npu-smi 23.0.0                    Version: 23.0.0                                  |
# +------------------------------------------------------------------------------------+
# | NPU  Name       Health   Power(W)   Temp(C)    Hugepages-Usage(page)  |
# | Chip  Device    Bus-Id   AICore(%)  Memory-Usage(MB)                   |
# +===================================================================================+
# | 0     910A      OK       65.0       42         0    / 0                 |
# | 0     0         0000:7B:00.0   0    0     / 32768                      |
# +------------------------------------------------------------------------------------+

# 检查CANN版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.info

# 检查MindSpore版本
python -c "import mindspore; print(mindspore.__version__)"
```

---

## 二、环境安装配置

### 2.1 安装CANN工具包

```bash
# 1. 下载CANN安装包 (从华为开发者网站)
# https://www.hiascend.com/software/cann/community

# 2. 安装CANN (以6.0.0为例)
chmod +x Ascend-cann-toolkit_6.0.0_linux-x86_64.run
./Ascend-cann-toolkit_6.0.0_linux-x86_64.run --install

# 3. 配置环境变量
echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh' >> ~/.bashrc
source ~/.bashrc

# 4. 验证安装
npu-smi info
```

### 2.2 创建Python环境

```bash
# 创建conda环境
conda create -n cra_srgan_ascend python=3.8 -y
conda activate cra_srgan_ascend

# 设置pip源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.3 安装MindSpore Ascend版本

```bash
# 安装MindSpore Ascend版本
pip install mindspore-ascend==2.2.0

# 或者使用华为镜像源
pip install mindspore-ascend==2.2.0 -i https://repo.huaweicloud.com/repository/pypi/simple/

# 验证安装
python -c "
import mindspore
from mindspore import context
context.set_context(device_target='Ascend')
print('MindSpore版本:', mindspore.__version__)
print('设备:', context.get_context('device_target'))
"
```

### 2.4 安装项目依赖

```bash
# 进入项目目录
cd CRA_SRGAN_Fusion

# 安装依赖
pip install -r requirements.txt

# 安装额外依赖
pip install opencv-python-headless albumentations
pip install pydicom nibabel  # 医学影像

# 验证环境
python -c "
import mindspore as ms
from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=0)
print('昇腾环境配置成功!')
print('MindSpore:', ms.__version__)
"
```

### 2.5 昇腾特定环境变量

```bash
# 添加到 ~/.bashrc
export ASCEND_HOME=/usr/local/Ascend
export PATH=$ASCEND_HOME/ascend-toolkit/latest/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_HOME/ascend-toolkit/latest/python/site-packages:$PYTHONPATH

# 设置日志级别 (可选,减少日志输出)
export GLOG_v=2
export ASCEND_GLOBAL_LOG_LEVEL=3

# 启用性能模式 (可选)
export ASCEND_SLOG_PRINT_TO_STDOUT=0

source ~/.bashrc
```

---

## 三、数据集准备

### 3.1 创建目录结构

```bash
cd CRA_SRGAN_Fusion

# 创建数据目录
mkdir -p datasets/cultural_relics/{train,val,test}/{images,masks}
mkdir -p datasets/medical/{train,val,test}/{images,masks}

# 创建输出目录
mkdir -p outputs checkpoints/{cultural_relics,medical} logs
```

### 3.2 数据准备

#### 自动准备 (推荐)

```bash
# 文物修复数据
python scripts/prepare_data.py \
    --mode cultural_relics \
    --input_dir /path/to/raw_images \
    --output_dir ./datasets/cultural_relics \
    --target_size 512 \
    --masks_per_image 3 \
    --split_ratio 0.9

# 医学影像数据
python scripts/prepare_data.py \
    --mode medical \
    --input_dir /path/to/medical_images \
    --output_dir ./datasets/medical \
    --target_size 512 \
    --masks_per_image 2 \
    --grayscale
```

#### 手动准备

```bash
# 将图像放入对应目录
datasets/cultural_relics/train/
├── images/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
└── masks/
    ├── 000001.png  # 白色=破损区域
    ├── 000002.png
    └── ...
```

### 3.3 数据验证

```bash
python -c "
from glob import glob
train_imgs = glob('./datasets/cultural_relics/train/images/*')
train_masks = glob('./datasets/cultural_relics/train/masks/*')
print(f'训练图像: {len(train_imgs)}')
print(f'训练掩码: {len(train_masks)}')
print('数据准备完成!' if len(train_imgs) > 0 else '警告: 未找到数据!')
"
```

---

## 四、模型训练

### 4.1 单卡训练

#### 文物修复训练

```bash
# 设置设备
export DEVICE_ID=0

# 开始训练
python scripts/train_cultural_relics.py \
    --image_dir ./datasets/cultural_relics/train/images \
    --mask_dir ./datasets/cultural_relics/train/masks \
    --batch_size 4 \
    --epochs 300 \
    --lr 0.0001 \
    --device Ascend \
    --device_id 0 \
    --save_dir ./checkpoints/cultural_relics \
    --log_interval 50 \
    --save_interval 20
```

#### 医学影像训练

```bash
# 设置设备
export DEVICE_ID=0

# 开始训练
python scripts/train_medical.py \
    --image_dir ./datasets/medical/train/images \
    --mask_dir ./datasets/medical/train/masks \
    --batch_size 4 \
    --epochs 200 \
    --lr 0.00005 \
    --device Ascend \
    --device_id 0 \
    --save_dir ./checkpoints/medical
```

### 4.2 多卡分布式训练

#### 方式1: 使用HCCL (华为集合通信库)

```bash
# 创建rank_table配置文件
# 8卡训练配置示例: rank_table_8pcs.json
cat > rank_table_8pcs.json << 'EOF'
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "localhost",
            "device": [
                {"device_id": "0", "device_ip": "192.168.100.101", "rank_id": "0"},
                {"device_id": "1", "device_ip": "192.168.100.102", "rank_id": "1"},
                {"device_id": "2", "device_ip": "192.168.100.103", "rank_id": "2"},
                {"device_id": "3", "device_ip": "192.168.100.104", "rank_id": "3"},
                {"device_id": "4", "device_ip": "192.168.100.105", "rank_id": "4"},
                {"device_id": "5", "device_ip": "192.168.100.106", "rank_id": "5"},
                {"device_id": "6", "device_ip": "192.168.100.107", "rank_id": "6"},
                {"device_id": "7", "device_ip": "192.168.100.108", "rank_id": "7"}
            ]
        }
    ],
    "status": "completed"
}
EOF
```

#### 方式2: 使用分布式启动脚本

```bash
# 创建分布式训练脚本
cat > scripts/run_distribute_train_ascend.sh << 'EOF'
#!/bin/bash
# 昇腾分布式训练脚本

export RANK_TABLE_FILE=$1
export DEVICE_NUM=$2
export RANK_SIZE=$2

TRAIN_IMAGE_DIR=$3
TRAIN_MASK_DIR=$4
EPOCHS=${5:-300}
BATCH_SIZE=${6:-4}

echo "=========================================="
echo "开始昇腾分布式训练"
echo "设备数量: $DEVICE_NUM"
echo "=========================================="

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
    export DEVICE_ID=$i
    export RANK_ID=$i
    
    echo "启动设备 $i ..."
    
    python train.py \
        --train_image_dir $TRAIN_IMAGE_DIR \
        --train_mask_dir $TRAIN_MASK_DIR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --device_target Ascend \
        --device_id $i \
        --run_distribute True \
        --device_num $DEVICE_NUM > train_device_$i.log 2>&1 &
done

echo "所有训练任务已启动"
echo "查看日志: tail -f train_device_0.log"
wait
EOF

chmod +x scripts/run_distribute_train_ascend.sh

# 执行8卡训练
./scripts/run_distribute_train_ascend.sh \
    rank_table_8pcs.json \
    8 \
    ./datasets/cultural_relics/train/images \
    ./datasets/cultural_relics/train/masks \
    300 \
    4
```

### 4.3 使用ModelArts云端训练

```bash
# 1. 上传数据到OBS
obsutil cp -r ./datasets obs://your-bucket/cra_srgan/datasets/

# 2. 上传代码到OBS
obsutil cp -r ./CRA_SRGAN_Fusion obs://your-bucket/cra_srgan/code/

# 3. 在ModelArts创建训练任务
# - 选择Ascend芯片
# - 选择MindSpore框架
# - 配置数据路径和代码路径
# - 启动训练
```

### 4.4 断点续训

```bash
# 从检查点继续训练
python scripts/train_cultural_relics.py \
    --image_dir ./datasets/cultural_relics/train/images \
    --mask_dir ./datasets/cultural_relics/train/masks \
    --resume ./checkpoints/cultural_relics/generator_epoch100.ckpt \
    --epochs 300 \
    --device Ascend \
    --device_id 0
```

### 4.5 监控训练

```bash
# 查看NPU使用情况
watch -n 1 npu-smi info

# 查看训练日志
tail -f ./checkpoints/cultural_relics/logs/train_log.txt

# TensorBoard (如果启用)
tensorboard --logdir=./logs --port=6006
```

---

## 五、模型推理

### 5.1 单卡推理

```bash
# 设置设备
export DEVICE_ID=0

# 单张图像推理
python infer.py \
    --input ./test_images/damaged.jpg \
    --mask ./test_images/mask.png \
    --output ./outputs/restored.png \
    --checkpoint ./checkpoints/cultural_relics/best_generator.ckpt \
    --device Ascend \
    --device_id 0
```

### 5.2 批量推理

```bash
# 批量处理
python infer.py \
    --input ./test_images/batch/ \
    --mask ./test_images/masks/ \
    --output ./outputs/batch/ \
    --checkpoint ./checkpoints/cultural_relics/best_generator.ckpt \
    --device Ascend \
    --batch_size 8
```

### 5.3 高分辨率分块推理

```bash
# 4K/8K图像分块推理
python infer.py \
    --input ./test_images/high_res.jpg \
    --mask ./test_images/high_res_mask.png \
    --output ./outputs/high_res_restored.png \
    --checkpoint ./checkpoints/cultural_relics/best_generator.ckpt \
    --device Ascend \
    --mode 8k \
    --tile_size 512 \
    --tile_overlap 64
```

### 5.4 Python API推理 (昇腾版)

```python
# inference_ascend.py
import os
import cv2
import numpy as np
import mindspore as ms
from mindspore import Tensor, context, load_checkpoint, load_param_into_net

# 设置昇腾环境
context.set_context(
    mode=context.GRAPH_MODE,
    device_target='Ascend',
    device_id=0
)

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


def inpaint_image(model, image_path, mask_path, output_path):
    """执行修复"""
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    
    # 预处理
    img = cv2.resize(img, (512, 512))
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (512, 512))
    
    img = img.astype(np.float32) / 127.5 - 1
    mask = mask.astype(np.float32) / 255.0
    
    img_tensor = Tensor(img.transpose(2, 0, 1)[np.newaxis, ...], ms.float32)
    mask_tensor = Tensor(mask[np.newaxis, np.newaxis, ...], ms.float32)
    
    # 创建输入
    x = img_tensor * (1 - mask_tensor)
    
    # 推理
    _, refine_out, _, _, _ = model(x, mask_tensor)
    
    # 后处理
    result = refine_out.asnumpy()[0].transpose(1, 2, 0)
    result = ((result + 1) * 127.5).clip(0, 255).astype(np.uint8)
    result = cv2.resize(result, (orig_w, orig_h))
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, result)
    print(f"保存到: {output_path}")


# 使用示例
if __name__ == '__main__':
    model = load_model('./checkpoints/cultural_relics/best_generator.ckpt')
    inpaint_image(model, 'damaged.jpg', 'mask.png', 'restored.png')
```

### 5.5 模型导出与部署

```bash
# 导出MindIR模型 (用于昇腾310推理)
python export_model.py \
    --checkpoint ./checkpoints/cultural_relics/best_generator.ckpt \
    --output ./exported_model/cra_srgan.mindir \
    --file_format MINDIR

# 导出ONNX模型
python export_model.py \
    --checkpoint ./checkpoints/cultural_relics/best_generator.ckpt \
    --output ./exported_model/cra_srgan.onnx \
    --file_format ONNX
```

---

## 六、评估与可视化

### 6.1 评估

```bash
# 全图评估
python scripts/evaluate.py \
    --gt_dir ./datasets/cultural_relics/test/images \
    --pred_dir ./outputs/test_results \
    --output ./evaluation_results.csv

# 修复区域评估
python scripts/evaluate.py \
    --gt_dir ./datasets/cultural_relics/test/images \
    --pred_dir ./outputs/test_results \
    --mask_dir ./datasets/cultural_relics/test/masks \
    --mode inpainting \
    --output ./inpainting_evaluation.csv
```

### 6.2 可视化

```bash
# 单张对比
python scripts/visualize.py \
    --mode single \
    --original ./test_images/original.jpg \
    --mask ./test_images/mask.png \
    --restored ./outputs/restored.png \
    --output ./comparison.png

# 批量对比图
python scripts/visualize.py \
    --mode batch \
    --original ./datasets/cultural_relics/test/images \
    --mask ./datasets/cultural_relics/test/masks \
    --restored ./outputs/test_results \
    --output ./visualization \
    --num_samples 20
```

---

## 七、性能优化

### 7.1 昇腾专属优化

```python
# 启用图模式和自动优化
from mindspore import context

context.set_context(
    mode=context.GRAPH_MODE,       # 图模式,性能更好
    device_target='Ascend',
    device_id=0,
    enable_graph_kernel=True,      # 启用图算融合
    graph_kernel_flags="--enable_cluster_ops=MatMul"
)
```

### 7.2 混合精度训练

```python
from mindspore import context
from mindspore.train.loss_scale_manager import FixedLossScaleManager

# 启用混合精度
context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

# 配置loss scale
loss_scale_manager = FixedLossScaleManager(loss_scale=1024, drop_overflow_update=False)
```

### 7.3 数据流水线优化

```python
import mindspore.dataset as ds

# 优化数据加载
dataset = ds.GeneratorDataset(
    source=your_dataset,
    column_names=['image', 'mask'],
    shuffle=True,
    num_parallel_workers=8,         # 并行工作线程
    python_multiprocessing=True     # 使用多进程
).batch(batch_size, drop_remainder=True)

# 预取数据
dataset = dataset.prefetch(buffer_size=4)
```

### 7.4 算子优化建议

```yaml
# 昇腾优化配置
optimization:
  # 使用昇腾优化的算子
  use_ascend_ops: true
  
  # 图算融合
  enable_graph_kernel: true
  
  # 内存优化
  memory_optimization: true
  
  # 自动并行
  auto_parallel: true
```

---

## 八、常见问题

### Q1: NPU设备不可见

```bash
# 检查驱动
npu-smi info

# 如果没有输出,重新安装驱动
/usr/local/Ascend/driver/script/npu_install.sh

# 检查CANN版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.info
```

### Q2: HCCL初始化失败

```bash
# 检查rank_table配置
cat rank_table_8pcs.json

# 检查网络连通性
ping 192.168.100.101

# 设置环境变量
export HCCL_CONNECT_TIMEOUT=6000
```

### Q3: 内存不足

```bash
# 方案1: 减小batch_size
--batch_size 2

# 方案2: 启用内存优化
export ASCEND_MEMPOOL_BLOCK_SIZE=256

# 方案3: 减小模型输入尺寸
```

### Q4: 训练速度慢

```bash
# 1. 启用图算融合
context.set_context(enable_graph_kernel=True)

# 2. 优化数据加载
num_parallel_workers=8

# 3. 使用多卡训练
./scripts/run_distribute_train_ascend.sh
```

### Q5: 模型导出失败

```bash
# 检查checkpoint格式
python -c "
from mindspore import load_checkpoint
params = load_checkpoint('your_model.ckpt')
print(f'参数数量: {len(params)}')
"

# 重新导出
python export_model.py --checkpoint model.ckpt --file_format MINDIR
```

---

## 附录: 完整命令汇总

```bash
# ==================== 环境配置 ====================
# 安装CANN
./Ascend-cann-toolkit_6.0.0_linux-x86_64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装MindSpore
pip install mindspore-ascend==2.2.0

# ==================== 数据准备 ====================
# 文物数据
python scripts/prepare_data.py --mode cultural_relics --input_dir /path/to/images --output_dir ./datasets/cultural_relics

# 医学数据
python scripts/prepare_data.py --mode medical --input_dir /path/to/medical --output_dir ./datasets/medical --grayscale

# ==================== 单卡训练 ====================
# 文物修复
python scripts/train_cultural_relics.py --image_dir ./datasets/cultural_relics/train/images --mask_dir ./datasets/cultural_relics/train/masks --epochs 300 --device Ascend

# 医学影像
python scripts/train_medical.py --image_dir ./datasets/medical/train/images --mask_dir ./datasets/medical/train/masks --epochs 200 --device Ascend

# ==================== 多卡训练 ====================
./scripts/run_distribute_train_ascend.sh rank_table_8pcs.json 8 ./datasets/cultural_relics/train/images ./datasets/cultural_relics/train/masks 300 4

# ==================== 推理 ====================
python infer.py --input test.jpg --mask mask.png --output result.png --checkpoint ./checkpoints/cultural_relics/best_generator.ckpt --device Ascend
```

