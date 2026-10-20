# BloomAI Flowers102 图像分类项目

## 项目简介

本项目使用 PyTorch 对真实 Flowers102 花卉数据集进行 102 类图像分类。

项目包含：

- NumPy、Pandas、Matplotlib 真实数据分析
- 手写小型 ResNet
- ImageNet 预训练 ResNet18 迁移学习
- 模型保存、测试集评估和单张图片预测
- Git 分阶段提交开发历史

## 环境

- Python 3.9.13
- PyTorch 2.4.1+cpu
- TorchVision 0.19.1+cpu
- Windows CPU 运行

## 数据集

Flowers102 数据集保存在本地 `data/` 目录：

- 训练集：1020 张图片
- 测试集：6149 张图片
- 类别：102 类

数据集不提交到 Git 仓库。

## 从零运行

进入项目目录并激活虚拟环境：

```powershell
cd C:\Users\LENOVO\BloomAI
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
pip install -r requirements.txt
```

首次训练模型：

```powershell
python src\train_flowers102_transfer.py
```

训练手写 ResNet：

```powershell
python src\train_flowers102.py
```

评估测试集：

```powershell
python src\evaluate_flowers102.py
```

评估手写 ResNet：

```powershell
python src\evaluate_manual_resnet.py
```

分析真实数据：

```powershell
python src\analyze_flowers.py
```

预测单张图片：

```powershell
python src\predict_image.py --image "C:\path\to\flower.jpg"
```

训练脚本会自动创建 `outputs` 目录，并保存：

```text
outputs/flowers102_resnet18_transfer.pth
```

数据分析脚本会生成：

```text
outputs/flowers102_class_statistics.csv
outputs/flowers102_class_distribution.png
```

预测程序会输出图片路径、类别编号、英文类别名称、中文译名和置信度。名称文件分别保存在 `src/flowers102_labels.json` 和 `src/flowers102_labels_zh.json`，程序会根据模型输出的类别编号读取对应名称。
