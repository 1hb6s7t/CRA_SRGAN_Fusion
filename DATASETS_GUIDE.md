# CRA-SRGAN 推荐数据集指南

---

## 一、文物修复方向数据集

### 1.1 推荐数据集列表

| 数据集 | 规模 | 类型 | 下载方式 |
|--------|------|------|----------|
| **Places365-Standard** | 180万张 | 场景图像 | 官网下载 |
| **CelebA-HQ** | 3万张 | 高清人脸 | 官网/Kaggle |
| **DIV2K** | 1000张 | 高清自然图像 | 官网下载 |
| **Paris StreetView** | 15000张 | 建筑街景 | 申请下载 |
| **敦煌壁画数据集** | 数千张 | 敦煌壁画 | 研究申请 |
| **ImageNet** | 120万张 | 通用图像 | 官网申请 |

---

### 1.2 详细数据集信息

#### 📦 Places365-Standard (强烈推荐)

**说明**: 包含365类场景的大规模数据集，适合训练通用图像修复模型

**下载地址**: http://places2.csail.mit.edu/download.html

```bash
# 下载命令 (选择Standard版本)
# 训练集 (约24GB)
wget http://data.csail.mit.edu/places/places365/train_large_places365standard.tar

# 验证集 (约2GB)  
wget http://data.csail.mit.edu/places/places365/val_large.tar

# 解压
tar -xvf train_large_places365standard.tar -C ./datasets/places365/
tar -xvf val_large.tar -C ./datasets/places365/
```

**处理命令**:
```bash
python scripts/prepare_data.py \
    --mode cultural_relics \
    --input_dir ./datasets/places365/train_large \
    --output_dir ./datasets/cultural_relics \
    --target_size 512
```

---

#### 📦 CelebA-HQ (人像/肖像画修复)

**说明**: 高清人脸数据集，适合训练人像修复和肖像画修复

**下载方式1 - Kaggle**:
```bash
# 需要Kaggle账号和API
pip install kaggle
kaggle datasets download -d lamsimon/celebahq
unzip celebahq.zip -d ./datasets/celebahq/
```

**下载方式2 - Google Drive**:
- 链接: https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs

**处理命令**:
```bash
python scripts/prepare_data.py \
    --mode cultural_relics \
    --input_dir ./datasets/celebahq/images \
    --output_dir ./datasets/portrait \
    --target_size 512
```

---

#### 📦 DIV2K (高清图像超分辨率)

**说明**: 专门用于超分辨率的高质量数据集，包含2K分辨率图像

**下载地址**: https://data.vision.ee.ethz.ch/cvl/DIV2K/

```bash
# 训练集HR图像
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip

# 验证集HR图像
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

# 解压
unzip DIV2K_train_HR.zip -d ./datasets/DIV2K/
unzip DIV2K_valid_HR.zip -d ./datasets/DIV2K/
```

**处理命令**:
```bash
python scripts/prepare_data.py \
    --mode cultural_relics \
    --input_dir ./datasets/DIV2K/DIV2K_train_HR \
    --output_dir ./datasets/div2k_inpaint \
    --target_size 512
```

---

#### 📦 Paris StreetView (建筑修复)

**说明**: 巴黎街景数据集，适合古建筑修复

**申请地址**: https://github.com/pathak22/context-encoder

**处理命令**:
```bash
python scripts/prepare_data.py \
    --mode cultural_relics \
    --input_dir ./datasets/paris_streetview \
    --output_dir ./datasets/architecture \
    --target_size 512
```

---

#### 📦 中国文物/敦煌数据集 (研究用途)

**敦煌壁画数据集申请**:
- 敦煌研究院: http://www.dha.ac.cn/
- 联系邮箱申请研究使用

**替代方案 - 公开古画数据**:
```bash
# 从网络收集中国古画图像
# 故宫博物院数字文物库: https://digicol.dpm.org.cn/
# 台北故宫: https://theme.npm.edu.tw/opendata/
```

---

### 1.3 掩码数据集

#### 📦 Irregular Mask Dataset (不规则掩码)

**下载地址**: https://nv-adlr.github.io/publication/partialconv-inpainting

```bash
# 下载不规则掩码
wget https://nv-adlr.github.io/files/irregular_masks.tar

tar -xvf irregular_masks.tar -d ./datasets/masks/
```

**目录结构**:
```
datasets/masks/
├── irregular_mask/
│   ├── testing_mask_dataset/
│   └── training_mask_dataset/
└── brush_mask/
```

---

## 二、医学影像方向数据集

### 2.1 推荐数据集列表

| 数据集 | 规模 | 类型 | 下载方式 |
|--------|------|------|----------|
| **ChestX-ray14** | 112,120张 | 胸部X光 | NIH官网 |
| **ISIC 2018** | 10,015张 | 皮肤病变 | ISIC官网 |
| **BraTS 2020** | 369例 | 脑部MRI | CBICA申请 |
| **DRIVE** | 40张 | 视网膜血管 | 官网下载 |
| **LUNA16** | 888例CT | 肺部CT | 官网下载 |
| **COVID-CT** | 746张 | COVID-19 CT | GitHub |

---

### 2.2 详细数据集信息

#### 📦 ChestX-ray14 (强烈推荐 - 胸部X光)

**说明**: NIH发布的最大胸部X光数据集，包含14种疾病标签

**下载地址**: https://nihcc.app.box.com/v/ChestXray-NIHCC

```bash
# 下载脚本 (需要分批下载)
# 文件列表: images_001.tar.gz ~ images_012.tar.gz (约45GB)

# 方法1: 使用wget批量下载
for i in $(seq -w 1 12); do
    wget https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkd-k.tar.gz -O images_0${i}.tar.gz
done

# 方法2: 使用Box直接下载
# 访问上述链接，手动下载

# 解压
for f in images_*.tar.gz; do tar -xzf $f -C ./datasets/chestxray/; done
```

**处理命令**:
```bash
python scripts/prepare_data.py \
    --mode medical \
    --input_dir ./datasets/chestxray/images \
    --output_dir ./datasets/medical_xray \
    --target_size 512 \
    --grayscale
```

---

#### 📦 ISIC 2018 (皮肤病变)

**说明**: 皮肤病变分割数据集，高质量皮肤科图像

**下载地址**: https://challenge.isic-archive.com/data/#2018

```bash
# 训练图像
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip

# 解压
unzip ISIC2018_Task1-2_Training_Input.zip -d ./datasets/ISIC2018/
```

**处理命令**:
```bash
python scripts/prepare_data.py \
    --mode medical \
    --input_dir ./datasets/ISIC2018/ISIC2018_Task1-2_Training_Input \
    --output_dir ./datasets/medical_skin \
    --target_size 512
```

---

#### 📦 DRIVE (视网膜血管)

**说明**: 视网膜血管分割数据集

**下载地址**: https://drive.grand-challenge.org/

```bash
# 需要注册账号下载
# 下载后解压到 ./datasets/DRIVE/
```

**处理命令**:
```bash
python scripts/prepare_data.py \
    --mode medical \
    --input_dir ./datasets/DRIVE/training/images \
    --output_dir ./datasets/medical_retina \
    --target_size 512
```

---

#### 📦 BraTS 2020 (脑肿瘤MRI)

**说明**: 脑肿瘤MRI分割挑战赛数据集

**申请地址**: https://www.med.upenn.edu/cbica/brats2020/registration.html

```bash
# 需要注册并申请访问权限
# 数据为NIfTI格式 (.nii.gz)

# 安装nibabel处理NIfTI
pip install nibabel

# 处理脚本
python scripts/process_nifti.py \
    --input_dir ./datasets/BraTS2020 \
    --output_dir ./datasets/medical_brain
```

---

#### 📦 COVID-CT (COVID-19 CT)

**说明**: COVID-19肺部CT数据集

**下载地址**: https://github.com/UCSD-AI4H/COVID-CT

```bash
# 克隆仓库
git clone https://github.com/UCSD-AI4H/COVID-CT.git ./datasets/COVID-CT

# 图像在 ./datasets/COVID-CT/Images-processed/
```

**处理命令**:
```bash
python scripts/prepare_data.py \
    --mode medical \
    --input_dir ./datasets/COVID-CT/Images-processed/CT_COVID \
    --output_dir ./datasets/medical_covid \
    --target_size 512 \
    --grayscale
```

---

#### 📦 LUNA16 (肺部结节检测)

**说明**: 肺结节检测挑战赛数据集

**下载地址**: https://luna16.grand-challenge.org/Download/

```bash
# 需要注册下载
# 数据为.mhd格式

pip install SimpleITK
# 使用SimpleITK读取处理
```

---

## 三、数据集使用建议

### 3.1 文物修复推荐组合

```bash
# 推荐方案: Places365 + DIV2K + CelebA-HQ混合训练
# 总计: ~5万张高质量图像

# 1. 准备Places365子集 (选取相关类别)
python scripts/prepare_data.py --mode cultural_relics --input_dir ./datasets/places365/train_large --output_dir ./datasets/cultural_relics --target_size 512

# 2. 准备DIV2K
python scripts/prepare_data.py --mode cultural_relics --input_dir ./datasets/DIV2K/DIV2K_train_HR --output_dir ./datasets/cultural_relics_div2k --target_size 512

# 3. 合并数据集
mkdir -p ./datasets/cultural_combined/train/images
mkdir -p ./datasets/cultural_combined/train/masks
cp ./datasets/cultural_relics/train/images/* ./datasets/cultural_combined/train/images/
cp ./datasets/cultural_relics_div2k/train/images/* ./datasets/cultural_combined/train/images/
```

### 3.2 医学影像推荐组合

```bash
# 推荐方案: ChestX-ray14 + ISIC2018 混合训练
# 或针对特定领域单独训练

# X光专用模型
python scripts/train_medical.py \
    --image_dir ./datasets/medical_xray/train/images \
    --mask_dir ./datasets/medical_xray/train/masks \
    --save_dir ./checkpoints/medical_xray

# 皮肤专用模型  
python scripts/train_medical.py \
    --image_dir ./datasets/medical_skin/train/images \
    --mask_dir ./datasets/medical_skin/train/masks \
    --save_dir ./checkpoints/medical_skin
```

---

## 四、数据集下载汇总表

### 4.1 文物修复数据集下载链接

| 数据集 | 下载链接 |
|--------|----------|
| Places365 | http://places2.csail.mit.edu/download.html |
| CelebA-HQ | https://www.kaggle.com/datasets/lamsimon/celebahq |
| DIV2K | https://data.vision.ee.ethz.ch/cvl/DIV2K/ |
| Irregular Masks | https://nv-adlr.github.io/publication/partialconv-inpainting |

### 4.2 医学影像数据集下载链接

| 数据集 | 下载链接 |
|--------|----------|
| ChestX-ray14 | https://nihcc.app.box.com/v/ChestXray-NIHCC |
| ISIC 2018 | https://challenge.isic-archive.com/data/#2018 |
| DRIVE | https://drive.grand-challenge.org/ |
| BraTS 2020 | https://www.med.upenn.edu/cbica/brats2020/ |
| COVID-CT | https://github.com/UCSD-AI4H/COVID-CT |
| LUNA16 | https://luna16.grand-challenge.org/Download/ |

---

## 五、快速开始脚本

### 5.1 一键下载脚本

```bash
#!/bin/bash
# download_datasets.sh
# 数据集下载脚本

mkdir -p datasets

echo "===== 下载DIV2K数据集 ====="
wget -P datasets/ https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip datasets/DIV2K_train_HR.zip -d datasets/DIV2K/

echo "===== 下载不规则掩码 ====="
# wget掩码数据集

echo "===== 克隆COVID-CT数据集 ====="
git clone https://github.com/UCSD-AI4H/COVID-CT.git datasets/COVID-CT

echo "下载完成!"
```

### 5.2 数据准备脚本

```bash
#!/bin/bash
# prepare_all_data.sh

# 文物修复数据
python scripts/prepare_data.py \
    --mode cultural_relics \
    --input_dir ./datasets/DIV2K/DIV2K_train_HR \
    --output_dir ./datasets/cultural_relics \
    --target_size 512 \
    --masks_per_image 3

# 医学影像数据
python scripts/prepare_data.py \
    --mode medical \
    --input_dir ./datasets/COVID-CT/Images-processed/CT_COVID \
    --output_dir ./datasets/medical \
    --target_size 512 \
    --grayscale

echo "数据准备完成!"
```

---

## 六、注意事项

1. **版权问题**: 部分数据集仅限研究用途，商业使用需获取授权
2. **数据大小**: ChestX-ray14约45GB，下载需要时间
3. **存储空间**: 建议准备至少200GB存储空间
4. **数据格式**: 医学影像可能是DICOM/NIfTI格式，需要转换
5. **隐私保护**: 医学数据使用需遵守伦理规范

