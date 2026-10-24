# BloomAI TensorFlow Flowers 图像分类项目

## 项目简介

本项目使用 PyTorch 和手写改进版 ResNet，对 TensorFlow Flowers 数据集进行 5 类花卉分类。

类别为：

- daisy（雏菊）
- dandelion（蒲公英）
- roses（玫瑰）
- sunflowers（向日葵）
- tulips（郁金香）

项目包含真实数据分析、手写残差块、模型训练、训练曲线、独立测试和单张图片预测。

完整的项目背景、实现原理、运行流程和提交材料说明见：

```text
项目说明.md
```

## 数据集和划分

数据集由 `prepare_data.py` 从官方 TensorFlow Flowers 地址下载，原始数据共 3670 张图片。

运行 `create_custom_split.py` 后生成固定随机划分：

- 训练集：2576 张
- 验证集：547 张
- 测试集：547 张

三个集合互不重叠。数据集图片和训练权重不提交到 GitHub。

## 从零运行

```powershell
cd C:\Users\LENOVO\BloomAI
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\prepare_data.py
python src\create_custom_split.py
```

训练手写改进版 ResNet：

```powershell
python src\train_TensorFlowFlowers.py
```

训练完成后会生成：

```text
outputs/tf_flowers_resnet_improved.pth
outputs/tf_flowers_training_history.csv
outputs/tf_flowers_training_curves.png
```

评估独立测试集：

```powershell
python src\evaluate_manual_resnet.py
```

预测桌面上的单张图片：

```powershell
python src\predict_manual_image.py --image "C:\Users\LENOVO\Desktop\DSC_0979.jpg"
```

程序会输出前五个候选类别、英文名称、中文名称、类别编号和概率。

## 数据分析

```powershell
python src\analyze_flowers.py
```

脚本使用 NumPy、Pandas 和 Matplotlib，生成：

```text
outputs/tf_flowers_class_statistics.csv
outputs/tf_flowers_class_distribution.png
```

## 模型说明

模型由项目自行编写残差块和跳跃连接，最后输出 5 个类别。训练过程中保存每轮训练损失和验证集准确率，并保存验证集表现最好的模型。
