#!/bin/bash
# ============================================================
# 昇腾分布式训练启动脚本
# 
# 用法:
#   ./run_distribute_train_ascend.sh <rank_table> <device_num> <train_image_dir> <train_mask_dir> [epochs] [batch_size]
#
# 参数:
#   rank_table     - HCCL配置文件路径
#   device_num     - 使用的NPU数量 (1, 2, 4, 8)
#   train_image_dir - 训练图像目录
#   train_mask_dir  - 训练掩码目录
#   epochs         - 训练轮数 (默认: 300)
#   batch_size     - 批大小 (默认: 4)
#
# 示例:
#   ./run_distribute_train_ascend.sh rank_table_8pcs.json 8 ./datasets/train/images ./datasets/train/masks 300 4
# ============================================================

set -e

# 检查参数
if [ $# -lt 4 ]; then
    echo "用法: $0 <rank_table> <device_num> <train_image_dir> <train_mask_dir> [epochs] [batch_size]"
    exit 1
fi

# 解析参数
export RANK_TABLE_FILE=$1
export DEVICE_NUM=$2
export RANK_SIZE=$2
TRAIN_IMAGE_DIR=$3
TRAIN_MASK_DIR=$4
EPOCHS=${5:-300}
BATCH_SIZE=${6:-4}

# 获取脚本所在目录
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_DIR"

# 检查文件
if [ ! -f "$RANK_TABLE_FILE" ]; then
    echo "错误: rank_table文件不存在: $RANK_TABLE_FILE"
    exit 1
fi

if [ ! -d "$TRAIN_IMAGE_DIR" ]; then
    echo "错误: 训练图像目录不存在: $TRAIN_IMAGE_DIR"
    exit 1
fi

if [ ! -d "$TRAIN_MASK_DIR" ]; then
    echo "错误: 训练掩码目录不存在: $TRAIN_MASK_DIR"
    exit 1
fi

# 创建日志目录
LOG_DIR="./logs/distribute_train_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "          昇腾分布式训练"
echo "============================================================"
echo "RANK_TABLE: $RANK_TABLE_FILE"
echo "设备数量: $DEVICE_NUM"
echo "训练图像: $TRAIN_IMAGE_DIR"
echo "训练掩码: $TRAIN_MASK_DIR"
echo "训练轮数: $EPOCHS"
echo "批大小: $BATCH_SIZE"
echo "日志目录: $LOG_DIR"
echo "============================================================"

# 设置环境变量
export GLOG_v=2
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

# 清理之前的进程
echo "清理旧进程..."
pkill -f "python.*train.py" 2>/dev/null || true
sleep 2

# 启动训练
echo "启动分布式训练..."
for ((i = 0; i < ${DEVICE_NUM}; i++)); do
    export DEVICE_ID=$i
    export RANK_ID=$i
    
    echo "启动设备 $i (RANK_ID=$i)..."
    
    python train.py \
        --train_image_dir "$TRAIN_IMAGE_DIR" \
        --train_mask_dir "$TRAIN_MASK_DIR" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --device_target Ascend \
        --device_id $i \
        --run_distribute True \
        --device_num $DEVICE_NUM \
        --save_dir "./checkpoints/distribute_device_$i" \
        > "$LOG_DIR/device_$i.log" 2>&1 &
    
    echo "  PID: $!"
done

echo ""
echo "============================================================"
echo "所有训练任务已启动!"
echo "============================================================"
echo ""
echo "查看日志:"
echo "  tail -f $LOG_DIR/device_0.log"
echo ""
echo "查看所有设备日志:"
echo "  tail -f $LOG_DIR/device_*.log"
echo ""
echo "停止训练:"
echo "  pkill -f 'python.*train.py'"
echo ""

# 等待所有任务完成
wait

echo "训练完成!"

