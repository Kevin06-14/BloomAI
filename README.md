# BloomAI Flowers102 图像分类项目

## 项目简介

本项目使用 PyTorch 对真实 Flowers102 花卉数据集进行 102 类图像分类。

项目包含：

- NumPy、Pandas、Matplotlib 真实数据分析
- 手写改进版 ResNet
- 模型保存、测试集评估和单张图片预测
- Git 分阶段提交开发历史

## 实验结果

本项目以手写改进版 ResNet 为主要模型：

| 模型 | 说明 | 自定义测试集准确率 |
| --- | --- | ---: |
| ImprovedResNet | 自己编写残差块、跳跃连接和分类层，并进行训练优化 | 0.7593（75.93%） |

准确率来自 918 张互不重复的自定义测试图片。

## 环境

- Python 3.9.13
- PyTorch 2.4.1+cpu
- TorchVision 0.19.1+cpu
- Windows CPU 运行

## 数据集

项目使用真实 Flowers102 数据集，并在本地生成自定义划分：

- 训练集：5150 张图片
- 验证集：2121 张图片
- 自定义测试集：918 张图片
- 类别：102 类

三个集合互不重叠。数据集图片不提交到 GitHub；新电脑可运行 `prepare_data.py` 下载原始数据，再运行 `create_custom_split.py` 生成相同划分。

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

准备数据集：

```powershell
python src\prepare_data.py
```

这个脚本会自动下载并准备 Flowers102 训练集和测试集。网络正常时，新电脑只需要执行一次。

创建自定义数据划分：

```powershell
python src\create_custom_split.py
```

当前划分约为 5150 张训练图片、918 张测试图片和 2121 张验证图片。三个集合互不重叠。

训练改进版手写 ResNet：

```powershell
python src\train_flowers102.py
```

训练后会生成：

```text
outputs/flowers102_resnet_improved.pth
outputs/manual_improved_training_history.csv
outputs/manual_improved_training_curves.png
outputs/manual_training_history.csv
outputs/manual_training_curves.png
```

评估改进版手写 ResNet：

```powershell
python src\evaluate_manual_resnet.py
```

使用改进版手写 ResNet 预测单张图片：

```powershell
python src\predict_manual_image.py --image "C:\path\to\flower.jpg"
```

分析真实数据：

```powershell
python src\analyze_flowers.py
```

预测单张图片：

```powershell
python src\predict_image.py --image "C:\path\to\flower.jpg"
```

训练日志中的 `train_loss` 表示训练集损失，`val_accuracy` 表示每轮在验证集上的准确率。PNG 曲线用于直观看出损失变化和验证准确率变化。

数据分析脚本会生成：

```text
outputs/flowers102_class_statistics.csv
outputs/flowers102_class_distribution.png
```

预测程序会输出图片路径、类别编号、英文类别名称、中文译名和置信度。名称文件分别保存在 `src/flowers102_labels.json` 和 `src/flowers102_labels_zh.json`，程序会根据模型输出的类别编号读取对应名称。

## 推荐演示顺序

如果需要从头展示项目，可以按以下顺序运行：

```powershell
python src\analyze_flowers.py
python src\evaluate_manual_resnet.py
python src\predict_manual_image.py --image "C:\path\to\flower.jpg"
```

其中，训练命令只需要在模型文件不存在时运行；已有模型时直接评估即可。
