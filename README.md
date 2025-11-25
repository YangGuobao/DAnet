# DAnet: Dual Attention Network for Remote Sensing Change Detection

<div align="center">
  <img src="figures/structure.png" width="800" alt="DAnet Architecture"/>
</div>

<br/>

<div align="center">
    <a href="https://github.com/YangGuobao/DAnet">
        <img src="https://img.shields.io/badge/Paper-TGRS%202025-red.svg" alt="Paper">
    </a>
    <a href="https://github.com/YangGuobao/DAnet/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
    </a>
    <a href="https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.8%2B-ee4c2c.svg" alt="PyTorch">
    </a>
</div>

<br/>

<div align="center">
    <h3>
        <a href="#english">English</a> | <a href="#chinese">中文说明</a>
    </h3>
</div>

<br/>

<span id="english"></span>

## Introduction

**DAnet** (Dual Attention Network) is a robust deep learning framework designed for binary change detection in high-resolution remote sensing imagery. 

Unlike traditional methods, DAnet specifically addresses the challenges of **complex scenes** (e.g., mountainous areas, seasonal variations) by incorporating a **Dual Attention Mechanism** (Spatial & Channel) to refine feature representations and suppress pseudo-changes.

### Key Features
- **Dual Attention Mechanism**: Simultaneously captures long-range spatial dependencies and channel-wise feature correlations.
- **Robustness**: Optimized for complex environments with significant appearance differences between dual-temporal images.
- **Efficiency**: Achieves a superior balance between accuracy and computational cost.

## Requirements

The code has been tested on **Linux** with **Python 3.8** and **PyTorch 1.10**.

- Python 3.8+
- PyTorch 1.8+ 
- CUDA 11.0+
- Other dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Dataset Preparation

We follow the standard directory structure used in **LEVIR-CD**, **WHU-CD**, and other mainstream datasets.

Please organize your dataset as follows:

```text
Dataset_Root/
├── A/          # T1 images (Pre-event)
├── B/          # T2 images (Post-event)
├── label/      # Ground Truth (Binary: 0/255)
└── list/       # Data split files
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## Usage

### 1. Training
You can train the model using the following command. Adjust the `--data_dir` to your dataset path.

```bash
python train.py --data_dir /path/to/your/dataset --batch_size 8 --epochs 200 --lr 0.0001
```

### 2. Evaluation
To evaluate the model on the test set:

```bash
python test.py --weights ./checkpoints/best_model.pth --data_dir /path/to/your/dataset
```

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@article{DAnet2025,
  title={DAnet: Dual Attention Mechanism for Efficient Binary Change Detection in Remote Sensing Imagery},
  author={Yang, Guobao and et al.},
  journal={IEEE Transactions on Geoscience and Remote Sensing (TGRS)},
  year={2025},
  note={Accepted/Under Review}
}
```

## Acknowledgements
We appreciate the open-source community. Part of this code is inspired by [BIT](https://github.com/justchenhao/BIT_CD) and [ChangeFormer](https://github.com/wgcban/ChangeFormer).

<br/>
<br/>
<hr>
<br/>
<br/>

<span id="chinese"></span>

<div align="center">
    <h3><a href="#english">⬆️ Back to English</a></h3>
</div>

## 简介 (Introduction)

**DAnet** (Dual Attention Network) 是一个用于高分辨率遥感图像二值变化检测的鲁棒深度学习框架。

与传统方法不同，DAnet 专门针对**复杂场景**（如西部山地、季节性变化显著区域）的难点进行了优化。它引入了**双重注意力机制**（空间注意力与通道注意力），能够有效提取关键特征并抑制伪变化。

### 核心特性
- **双重注意力机制**：同时捕捉长距离的空间依赖关系和通道间的特征关联。
- **强鲁棒性**：针对双时相图像外观差异巨大的复杂环境进行了优化。
- **高效性**：在检测精度和计算成本之间取得了优异的平衡。

## 环境要求 (Requirements)

代码已在 **Linux** 环境下，使用 **Python 3.8** 和 **PyTorch 1.10** 完成测试。

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+
- 安装依赖库：
  ```bash
  pip install -r requirements.txt
  ```

## 数据准备 (Data Preparation)

我们采用与 **LEVIR-CD**、**WHU-CD** 等主流数据集一致的标准目录结构。

请按以下方式组织您的数据：

```text
Dataset_Root/
├── A/          # T1 时相图像 (变化前)
├── B/          # T2 时相图像 (变化后)
├── label/      # 真值标签 (二值图: 0 代表未变化, 255 代表变化)
└── list/       # 数据集划分文件
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## 使用说明 (Usage)

### 1. 训练 (Training)
使用以下命令启动训练。请将 `--data_dir` 修改为您的数据集路径。

```bash
python train.py --data_dir /path/to/your/dataset --batch_size 8 --epochs 200 --lr 0.0001
```

### 2. 评估 (Evaluation)
在测试集上评估模型性能：

```bash
python test.py --weights ./checkpoints/best_model.pth --data_dir /path/to/your/dataset
```

## 引用 (Citation)

如果您在研究中使用了本项目的代码，请引用我们的论文：

```bibtex
@article{DAnet2025,
  title={DAnet: Dual Attention Mechanism for Efficient Binary Change Detection in Remote Sensing Imagery},
  author={Yang, Guobao and et al.},
  journal={IEEE Transactions on Geoscience and Remote Sensing (TGRS)},
  year={2025},
  note={Accepted/Under Review}
}
```

## 许可 (License)
本项目基于 [Apache 2.0 License](LICENSE) 开源。
