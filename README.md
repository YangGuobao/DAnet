# DAnet
An efficient change detection network based on a Siamese architecture. It incorporates a difference enhancement module to robustly identify changes in high-resolution imagery while effectively reducing noise from seasonal variations.
# DAnet: Dual Attention Network for Remote Sensing Change Detection

<div align="center">
  <img src="figures/structure.png" width="800"/>
</div>

<div align="center">
    <a href="https://github.com/YangGuobao/DAnet">
        <img src="https://img.shields.io/badge/Paper-Arxiv-red.svg" alt="Paper">
    </a>
    <a href="https://github.com/YangGuobao/DAnet/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
    </a>
</div>

## Introduction

**DAnet** is a robust deep learning network designed for binary change detection in high-resolution remote sensing imagery. 

Key features include:
- **Dual Attention Mechanism**: Effectively captures both spatial and channel dependencies to refine feature representations.
- **Robustness in Complex Scenes**: Specifically optimized for challenging environments, such as mountainous areas with significant seasonal variations.
- **Efficient Implementation**: Built on PyTorch for ease of use and extension.

This repository contains the official PyTorch implementation of DAnet.

## Requirements

Please ensure your environment meets the following requirements:

- Python 3.8+
- PyTorch 1.8+ (Tested on 1.10.0)
- CUDA 11.0+
- Other dependencies:
  ```bash
  pip install -r requirements.txt
