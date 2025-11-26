# D&A-CDNet: “先解耦，后对齐”遥感变化检测网络

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
        <img src="https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg" alt="PyTorch">
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

现有的遥感变化检测方法经常受到伪变化（如季节变化、光照差异）的干扰。我们认为其根本原因在于特征的高度耦合。为了解决这个问题，我们提出了一种**“先解耦，后对齐”**的新范式：

* **特征解耦 (Feature Decoupling)**：通过特征解耦模块 (FDM) 将双时相特征显式分解为**变化无关 (Change-Invariant)** 和 **变化敏感 (Change-Sensitive)** 两个子空间。

* **非对称对齐 (Asymmetric Alignment)**：
    * 利用 **$L_{tc}$ (时序一致性损失)** 强制对齐变化无关特征，以学习时序不变性。
    * 利用 **$L_{cs}$ (对比分离损失)** 在变化区域推远变化敏感特征，在不变区域将其拉近，从而增强可分性。

该模型在 **WHU-CD, CDD 和 S2Looking** 等数据集上均取得了优异的性能。

## 环境要求 (Requirements)

代码已在 **Python 3.8** 和 **PyTorch 2.10** 环境下完成测试。

- Python 3.8+
- PyTorch 2.1+ 
- 其他依赖库：
  ```bash
  pip install -r requirements.txt
