# BloomAI Flowers102 图像分类项目

## 项目简介

本项目使用 PyTorch 对真实花卉图片进行分类，数据集为 Flowers102，共包含 102 个花卉类别。

项目完成了：

- Python 虚拟环境配置
- NumPy、Pandas、Matplotlib、PyTorch 环境配置
- Flowers102 真实数据集下载
- 小型 ResNet 模型实现
- ResNet18 迁移学习训练
- 模型保存与加载
- 测试集准确率评估
- Git 分阶段提交开发历史

## 环境

- Python 3.9.13
- PyTorch 2.4.1+cpu
- TorchVision 0.19.1+cpu
- Windows
- CPU 运行

## 数据集

使用 Flowers102 真实花卉数据集：

- 训练集：1020 张图片
- 测试集：6149 张图片
- 类别数量：102 类

数据集保存在本地 `data/` 目录中，不提交到 Git 仓库。

## 运行方法

进入项目目录：

```powershell
cd C:\Users\LENOVO\BloomAI