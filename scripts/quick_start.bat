@echo off
REM CRA-SRGAN 快速启动脚本 (Windows)
REM =====================================

echo ============================================
echo     CRA-SRGAN 图像修复与超高清化模型
echo ============================================
echo.

:MENU
echo 请选择操作:
echo   1. 准备文物修复数据集
echo   2. 准备医学影像数据集
echo   3. 训练文物修复模型
echo   4. 训练医学影像模型
echo   5. 推理 (单张图像)
echo   6. 退出
echo.

set /p choice=请输入选项 (1-6): 

if "%choice%"=="1" goto PREPARE_CULTURAL
if "%choice%"=="2" goto PREPARE_MEDICAL
if "%choice%"=="3" goto TRAIN_CULTURAL
if "%choice%"=="4" goto TRAIN_MEDICAL
if "%choice%"=="5" goto INFERENCE
if "%choice%"=="6" goto END

echo 无效选项,请重新输入
goto MENU

:PREPARE_CULTURAL
echo.
echo === 准备文物修复数据集 ===
set /p input_dir=请输入原始图像目录: 
set /p output_dir=请输入输出目录 (默认: ./datasets/cultural_relics): 
if "%output_dir%"=="" set output_dir=./datasets/cultural_relics

python scripts/prepare_data.py --mode cultural_relics --input_dir %input_dir% --output_dir %output_dir%
echo.
pause
goto MENU

:PREPARE_MEDICAL
echo.
echo === 准备医学影像数据集 ===
set /p input_dir=请输入原始图像目录: 
set /p output_dir=请输入输出目录 (默认: ./datasets/medical): 
if "%output_dir%"=="" set output_dir=./datasets/medical

python scripts/prepare_data.py --mode medical --input_dir %input_dir% --output_dir %output_dir% --grayscale
echo.
pause
goto MENU

:TRAIN_CULTURAL
echo.
echo === 训练文物修复模型 ===
set /p image_dir=请输入训练图像目录 (默认: ./datasets/cultural_relics/train/images): 
set /p mask_dir=请输入掩码目录 (默认: ./datasets/cultural_relics/train/masks): 
if "%image_dir%"=="" set image_dir=./datasets/cultural_relics/train/images
if "%mask_dir%"=="" set mask_dir=./datasets/cultural_relics/train/masks

python scripts/train_cultural_relics.py --image_dir %image_dir% --mask_dir %mask_dir% --epochs 300 --batch_size 4
echo.
pause
goto MENU

:TRAIN_MEDICAL
echo.
echo === 训练医学影像模型 ===
set /p image_dir=请输入训练图像目录 (默认: ./datasets/medical/train/images): 
set /p mask_dir=请输入掩码目录 (默认: ./datasets/medical/train/masks): 
if "%image_dir%"=="" set image_dir=./datasets/medical/train/images
if "%mask_dir%"=="" set mask_dir=./datasets/medical/train/masks

python scripts/train_medical.py --image_dir %image_dir% --mask_dir %mask_dir% --epochs 200 --batch_size 4
echo.
pause
goto MENU

:INFERENCE
echo.
echo === 推理 ===
set /p input_img=请输入图像路径: 
set /p mask_img=请输入掩码路径: 
set /p checkpoint=请输入模型权重路径: 
set /p output=请输入输出路径 (默认: ./outputs/result.png): 
if "%output%"=="" set output=./outputs/result.png

python infer.py --input %input_img% --mask %mask_img% --checkpoint %checkpoint% --output %output%
echo.
pause
goto MENU

:END
echo 感谢使用!
pause

