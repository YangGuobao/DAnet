# D&A-CDNet: Decouple-then-Align Change Detection Network

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
        English | <a href="README_CN.md">简体中文</a>
    </h3>
</div>

<br/>

## Introduction

This is the official implementation of **D&A-CDNet** (Decouple-then-Align Change Detection Network).

Existing remote sensing change detection methods are often disturbed by pseudo-changes (e.g., seasonal variations, lighting differences). We argue that the root cause lies in the high coupling of features. To address this, we propose a new paradigm of **"Decouple-then-Align"**:

* **Feature Decoupling**: Explicitly decomposes bi-temporal features into **Change-Invariant** and **Change-Sensitive** subspaces via a Feature Decoupling Module (FDM).

* **Asymmetric Alignment**:
    * Utilizes **$L_{tc}$ (Temporal Consistency Loss)** to forcibly align change-invariant features to learn temporal invariance.
    * Utilizes **$L_{cs}$ (Contrastive Separation Loss)** to push apart change-sensitive features in changed areas and pull them closer in unchanged areas, enhancing separability.

The model achieves superior performance on datasets such as **LEVIR-CD, WHU-CD, SYSU-CD, and DSIFN**.

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
@article{DACDNet2025,
  title={D&A-CDNet: Decouple-then-Align Change Detection Network for Remote Sensing},
  author={Yang, Guobao and et al.},
  journal={IEEE Transactions on Geoscience and Remote Sensing (TGRS)},
  year={2025},
  note={Accepted/Under Review}
}
```

## Acknowledgements
We appreciate the open-source community. Part of this code is inspired by [BIT](https://github.com/justchenhao/BIT_CD) and [ChangeFormer](https://github.com/wgcban/ChangeFormer).

## License
This project is licensed under the [Apache 2.0 License](LICENSE).
