# ViT CIFAR-10

基于 PyTorch 实现的 ViT-Tiny 图像分类项目，用于在 CIFAR-10 数据集上训练和验证 Vision Transformer 模型。

## 项目结构

```text
vit_cifar10/
├── data/                 # CIFAR-10 数据集目录，不建议提交到 Git
├── datasets/
│   ├── __init__.py
│   └── data.py           # 数据增强、数据集加载、DataLoader 构建
├── engine/
│   ├── __init__.py
│   ├── train.py          # 训练逻辑、验证逻辑、模型保存
│   └── evalute.py        # 测试集评估脚本
├── models/
│   ├── __init__.py
│   └── vit_tiny.py       # ViT-Tiny 模型定义
├── outputs/              # 训练输出目录，不建议提交到 Git
├── main.py               # 项目训练入口
├── .gitignore
└── README.md