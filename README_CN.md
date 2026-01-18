# Decouple then Align: A Disentangled Representation Learning Framework for Remote Sensing Change Detection
# (解耦对齐：基于解耦表征学习的遥感影像变化检测网络)

<div align="center">
  <img src="figures/structure.png" width="800" alt="D&A-CDNet 架构图"/>
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
        <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch">
    </a>
</div>

<br/>

<div align="center">
    <h3>
        <a href="README.md">English</a> | 简体中文
    </h3>
</div>

<br/>

## 📖 简介 (Introduction)

这是 **D&A-CDNet** (Decouple-then-Align Change Detection Network) 的官方实现代码。

在遥感变化检测 (RSCD) 任务中，现有的深度学习方法经常受到由季节变化、光照差异和成像条件引起的**伪变化**干扰。我们认为，其根本原因在于深层特征空间中，感兴趣的目标变化特征与背景上下文特征存在高度的语义纠缠。

为了解决这个问题，我们提出了一种全新的 **“先解耦，后对齐” (Decouple-then-Align)** 范式：

1.  **主动解耦 (Decouple)**：我们设计了 **自适应残差门控解耦 (ARGD)** 模块。利用双重注意力背景建模和非线性门控机制，动态滤除噪声，并将特征显式投影为 **变化无关 (Change-Invariant)** 和 **变化敏感 (Change-Sensitive)** 两个子空间。
2.  **显式对齐 (Align)**：我们提出了 **多维正交对齐策略** 来显式监督解耦过程：
    * **$L_{ortho}$ (特征正交性约束)**：强制变化无关与变化敏感子空间在几何上保持正交，防止信息混入。
    * **$L_{tc}$ (掩膜引导的时序一致性损失)**：在非变化区域强制对齐背景特征，而在变化区域放松约束。
    * **$L_{cs}$ (对比分离损失)**：在变化区域推远变化敏感特征，在不变区域将其拉近。

该方法在 **WHU-CD** ($F_1$ 96.04%) 和 **CDD** ($F_1$ 97.28%) 数据集上均取得了 SOTA 性能。

## 🚀 主要结果 (Main Results)

**WHU-CD 和 CDD 数据集上的定量对比**

| 方法 (Method) | 骨干网络 (Backbone) | 数据集 (Dataset) | 精确率 (Precision) | 召回率 (Recall) | F1 分数 | IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| D&A-CDNet | PVT v2-B1 | **WHU-CD** | 96.54 | 95.55 | **96.04** | **92.39** |
| D&A-CDNet | PVT v2-B1 | **CDD** | 97.68 | 96.89 | **97.28** | **94.71** |

> **注意**: 我们的模型在性能与效率之间取得了良好的平衡，参数量为 **46.21 M**，计算量为 **13.08 G** FLOPs (256x256 输入)。

## 🛠️ 环境要求 (Requirements)

代码已在 **Python 3.8+** 和 **PyTorch 2.0+** 环境下完成测试。

```bash
pip install -r requirements.txt
```

*核心依赖:* `torch`, `torchvision`, `timm`, `safetensors`, `thop` (可选，用于计算 FLOPs)。

## 📂 数据准备 (Dataset Preparation)

我们遵循 **WHU-CD** 和 **CDD** 数据集的标准目录结构。请按以下方式组织您的数据集：

```text
data/
 ├── WHU-CD
    ├── train/          
        ├── A/          # T1 时相图像 (变化前)
        ├── B/          # T2 时相图像 (变化后)
        ├── label/      # 真值标签 (0/255)
    ├── test/
        ├── A/ 
        ├── B/ 
        ├── label/            
    ├── val/
        ├── A/ 
        ├── B/ 
        ├── label/       
 ├── CDD
    ├── train/ ...
    ├── test/ ...
    ├── val/ ...
```

## ⚖️ 预训练权重 (Pretrained Weights)

我们使用在 ImageNet 上预训练的 **PVT v2-B1** 作为骨干网络。

| 模型 | 来源 | 路径 |
| :--- | :--- | :--- |
| **PVT v2-B1** | [官方发布](https://github.com/whai362/PVT) | 请将 `pvt_v2_b1_weights.safetensors` 放置在项目根目录下。 |

## ⚡ 使用说明 (Usage)

### 1. 训练 (Training)

使用论文中描述的超参数进行训练 ($L_{tc}=0.5, L_{cs}=0.5, L_{ortho}=0.1$)：

```bash
python train.py \
  --dataset_root ./data \
  --data_name CDD \
  --backbone pvt_v2_b1 \
  --pretrained True \
  --batchsize 8 \
  --trainsize 256 \
  --epoch 200 \
  --lr 0.0001 \
  --lambda_tc 0.5 \
  --lambda_cs 0.5 \
  --lambda_ortho 0.1 \
  --contrastive_margin 1.0
```

### 2. 评估 (Evaluation)

在测试集上评估模型性能：

```bash
python test.py \
  --dataset_root ./data \
  --data_name CDD \
  --backbone pvt_v2_b1 \
  --checkpoint_path ./checkpoints/CDD/DA_CDNet_Ortho_pvt_v2_b1_xxxx/best_model.pth
```

## 📝 引用 (Citation)

如果您在研究中使用了本项目的代码，请引用我们的论文：

```bibtex
@article{DACDNet2025,
  title={Decouple then Align: A Disentangled Representation Learning Framework for Remote Sensing Change Detection},
  author={Yang, Guobao and et al.},
  journal={IEEE Transactions on Geoscience and Remote Sensing (TGRS)},
  year={2025},
  note={Under Review}
}
```

## 🙏 致谢 (Acknowledgements)

感谢开源社区的贡献。本项目的代码部分参考了 [BIT](https://github.com/justchenhao/BIT_CD) 和 [ChangeFormer](https://github.com/wgcban/ChangeFormer)。

## 📄 许可 (License)

本项目基于 [Apache 2.0 License](LICENSE) 开源。
