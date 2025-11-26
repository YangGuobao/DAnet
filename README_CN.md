# D&A-CDNet: “先解耦，后对齐”遥感变化检测网络

<div align="center">
  <img src="figures/structure.png" width="800" alt="D&A-CDNet Architecture"/>
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
        <a href="README.md">English</a> | 简体中文
    </h3>
</div>

<br/>

## 简介 (Introduction)

这是 **D&A-CDNet** (Decouple-then-Align Change Detection Network) 的官方实现代码。

现有的遥感变化检测方法经常受到伪变化（如季节、光照差异）的干扰。我们认为其根本原因在于特征的高度耦合。为了解决这个问题，我们提出了一种**“先解耦，后对齐”**的新范式：

* **特征解耦 (Feature Decoupling)**: 通过特征解耦模块 (FDM) 将双时相特征显式分解为**变化无关 (Change-Invariant)** 和 **变化敏感 (Change-Sensitive)** 两个子空间。

* **非对称对齐 (Asymmetric Alignment)**:
    * 利用 $L_{tc}$ (时序一致性损失) 强制对齐变化无关特征，学习时序不变性。
    * 利用 $L_{cs}$ (对比分离损失) 在变化区域推远、不变区域拉近变化敏感特征，增强可分性。

该模型在 **LEVIR-CD, WHU-CD, SYSU-CD 和 DSIFN** 等数据集上均取得了优异的性能。

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
@article{DACDNet2025,
  title={D&A-CDNet: Decouple-then-Align Change Detection Network for Remote Sensing},
  author={Yang, Guobao and et al.},
  journal={IEEE Transactions on Geoscience and Remote Sensing (TGRS)},
  year={2025},
  note={Accepted/Under Review}
}
```

## 致谢 (Acknowledgements)
感谢开源社区的贡献。本项目的代码部分参考了 [BIT](https://github.com/justchenhao/BIT_CD) 和 [ChangeFormer](https://github.com/wgcban/ChangeFormer)。

## 许可 (License)
本项目基于 [Apache 2.0 License](LICENSE) 开源。
