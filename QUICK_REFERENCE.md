# CRA-SRGAN 快速参考卡

## 🚀 一键启动命令

### GPU版本

```bash
# 1️⃣ 安装环境
conda create -n cra python=3.8 -y && conda activate cra
pip install mindspore==2.2.0 -r requirements.txt

# 2️⃣ 准备数据 (文物修复)
python scripts/prepare_data.py --mode cultural_relics --input_dir /您的图像目录 --output_dir ./datasets/cultural_relics

# 3️⃣ 训练
python scripts/train_cultural_relics.py --image_dir ./datasets/cultural_relics/train/images --mask_dir ./datasets/cultural_relics/train/masks --device GPU

# 4️⃣ 推理
python infer.py --input 破损图.jpg --mask 掩码.png --output 结果.png --checkpoint ./checkpoints/cultural_relics/best_generator.ckpt --device GPU
```

### 昇腾版本

```bash
# 1️⃣ 安装环境
conda create -n cra python=3.8 -y && conda activate cra
pip install mindspore-ascend==2.2.0 -r requirements.txt

# 2️⃣ 准备数据
python scripts/prepare_data.py --mode cultural_relics --input_dir /您的图像目录 --output_dir ./datasets/cultural_relics

# 3️⃣ 训练
python scripts/train_cultural_relics.py --image_dir ./datasets/cultural_relics/train/images --mask_dir ./datasets/cultural_relics/train/masks --device Ascend

# 4️⃣ 推理
python infer.py --input 破损图.jpg --mask 掩码.png --output 结果.png --checkpoint ./checkpoints/cultural_relics/best_generator.ckpt --device Ascend
```

---

## 📋 常用命令速查

| 任务 | GPU命令 | 昇腾命令 |
|------|---------|----------|
| **训练文物** | `python scripts/train_cultural_relics.py --device GPU` | `python scripts/train_cultural_relics.py --device Ascend` |
| **训练医学** | `python scripts/train_medical.py --device GPU` | `python scripts/train_medical.py --device Ascend` |
| **推理** | `python infer.py --device GPU` | `python infer.py --device Ascend` |
| **评估** | `python scripts/evaluate.py` | `python scripts/evaluate.py` |
| **可视化** | `python scripts/visualize.py` | `python scripts/visualize.py` |

---

## ⚙️ 关键参数

### 训练参数

| 参数 | 文物修复推荐值 | 医学影像推荐值 | 说明 |
|------|----------------|----------------|------|
| `--batch_size` | 4 | 4 | 减小可降低显存 |
| `--lr` | 0.0001 | 0.00005 | 学习率 |
| `--epochs` | 300 | 200 | 训练轮数 |

### 损失权重 (配置文件)

| 参数 | 文物修复 | 医学影像 | 说明 |
|------|----------|----------|------|
| `l1_weight` | 1.0 | 2.0 | 像素精度 |
| `perceptual_weight` | 0.2 | 0.05 | 感知损失 |
| `style_weight` | 0.15 | 0.01 | 风格损失 |
| `adversarial_weight` | 0.0005 | 0.0001 | GAN损失 |
| `edge_weight` | 0.15 | 0.2 | 边缘损失 |

---

## 📂 目录结构

```
CRA_SRGAN_Fusion/
├── datasets/                 # 数据集
│   ├── cultural_relics/     # 文物数据
│   │   ├── train/{images,masks}
│   │   └── val/{images,masks}
│   └── medical/             # 医学数据
├── checkpoints/              # 模型权重
├── outputs/                  # 推理输出
├── configs/                  # 配置文件
├── scripts/                  # 工具脚本
└── src/                      # 源代码
```

---

## 🔧 常见问题快速解决

### 显存/内存不足
```bash
--batch_size 2  # 减小batch
```

### 训练Loss不降
```bash
--lr 0.00005  # 降低学习率
```

### 修复结果模糊
```yaml
# 修改配置文件
perceptual_weight: 0.3
edge_weight: 0.2
```

### 多卡训练 (昇腾)
```bash
# 生成配置
python scripts/generate_rank_table.py --device_num 8

# 启动训练
./scripts/run_distribute_train_ascend.sh rank_table.json 8 ./datasets/cultural_relics/train/images ./datasets/cultural_relics/train/masks
```

---

## 📊 评估指标说明

| 指标 | 全称 | 说明 | 越大越好 |
|------|------|------|----------|
| PSNR | 峰值信噪比 | 像素精度 | ✅ |
| SSIM | 结构相似性 | 结构保持 | ✅ |
| LPIPS | 感知相似性 | 感知质量 | ❌ |
| FID | Fréchet距离 | 真实度 | ❌ |

---

## 📝 文件说明

| 文件 | 说明 |
|------|------|
| `USER_MANUAL_GPU.md` | GPU完整手册 |
| `USER_MANUAL_ASCEND.md` | 昇腾完整手册 |
| `TUTORIAL_文物修复与医学影像.md` | 详细教程 |
| `TECHNICAL_REPORT.md` | 技术报告 |
| `configs/cultural_relics.yaml` | 文物配置 |
| `configs/medical.yaml` | 医学配置 |

