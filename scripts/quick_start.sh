#!/bin/bash
# CRA-SRGAN 快速启动脚本 (Linux/Mac)
# =====================================

echo "============================================"
echo "    CRA-SRGAN 图像修复与超高清化模型"
echo "============================================"
echo ""

show_menu() {
    echo "请选择操作:"
    echo "  1. 准备文物修复数据集"
    echo "  2. 准备医学影像数据集"
    echo "  3. 训练文物修复模型"
    echo "  4. 训练医学影像模型"
    echo "  5. 推理 (单张图像)"
    echo "  6. 退出"
    echo ""
}

while true; do
    show_menu
    read -p "请输入选项 (1-6): " choice
    
    case $choice in
        1)
            echo ""
            echo "=== 准备文物修复数据集 ==="
            read -p "请输入原始图像目录: " input_dir
            read -p "请输入输出目录 (默认: ./datasets/cultural_relics): " output_dir
            output_dir=${output_dir:-./datasets/cultural_relics}
            
            python scripts/prepare_data.py --mode cultural_relics --input_dir "$input_dir" --output_dir "$output_dir"
            echo ""
            read -p "按回车继续..."
            ;;
        2)
            echo ""
            echo "=== 准备医学影像数据集 ==="
            read -p "请输入原始图像目录: " input_dir
            read -p "请输入输出目录 (默认: ./datasets/medical): " output_dir
            output_dir=${output_dir:-./datasets/medical}
            
            python scripts/prepare_data.py --mode medical --input_dir "$input_dir" --output_dir "$output_dir" --grayscale
            echo ""
            read -p "按回车继续..."
            ;;
        3)
            echo ""
            echo "=== 训练文物修复模型 ==="
            read -p "请输入训练图像目录 (默认: ./datasets/cultural_relics/train/images): " image_dir
            read -p "请输入掩码目录 (默认: ./datasets/cultural_relics/train/masks): " mask_dir
            image_dir=${image_dir:-./datasets/cultural_relics/train/images}
            mask_dir=${mask_dir:-./datasets/cultural_relics/train/masks}
            
            python scripts/train_cultural_relics.py --image_dir "$image_dir" --mask_dir "$mask_dir" --epochs 300 --batch_size 4
            echo ""
            read -p "按回车继续..."
            ;;
        4)
            echo ""
            echo "=== 训练医学影像模型 ==="
            read -p "请输入训练图像目录 (默认: ./datasets/medical/train/images): " image_dir
            read -p "请输入掩码目录 (默认: ./datasets/medical/train/masks): " mask_dir
            image_dir=${image_dir:-./datasets/medical/train/images}
            mask_dir=${mask_dir:-./datasets/medical/train/masks}
            
            python scripts/train_medical.py --image_dir "$image_dir" --mask_dir "$mask_dir" --epochs 200 --batch_size 4
            echo ""
            read -p "按回车继续..."
            ;;
        5)
            echo ""
            echo "=== 推理 ==="
            read -p "请输入图像路径: " input_img
            read -p "请输入掩码路径: " mask_img
            read -p "请输入模型权重路径: " checkpoint
            read -p "请输入输出路径 (默认: ./outputs/result.png): " output
            output=${output:-./outputs/result.png}
            
            python infer.py --input "$input_img" --mask "$mask_img" --checkpoint "$checkpoint" --output "$output"
            echo ""
            read -p "按回车继续..."
            ;;
        6)
            echo "感谢使用!"
            exit 0
            ;;
        *)
            echo "无效选项,请重新输入"
            ;;
    esac
done

