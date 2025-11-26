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
        <img src="https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg" alt="PyTorch">
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

The model achieves superior performance on datasets such as **WHU-CD, CDD, and S2Looking**.

## Requirements

The code has been tested with **Python 3.8** and **PyTorch 2.10**.

- Python 3.8+
- PyTorch 2.1+ 
- Other dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Dataset Preparation

We follow the standard directory structure used in **WHU-CD, CDD, and S2Looking**, and other mainstream datasets.

Please organize your dataset as follows:

```text
data/
 ├── WHU-CD
    ├── train/         
        ├── A/          # T1 images (Pre-event)
        ├── B/          # T2 images (Post-event)
        ├── label/      # Ground Truth (Binary: 0/255)
    ├── test/
        ├── A/          # T1 images (Pre-event)
        ├── B/          # T2 images (Post-event)
        ├── label/      # Ground Truth (Binary: 0/255)          
    ├── val/
        ├── A/          # T1 images (Pre-event)
        ├── B/          # T2 images (Post-event)
        ├── label/      # Ground Truth (Binary: 0/255)      
 ├── CDD
    .
    .
    .
 └── S2Looking
```
## Pretrained Weights
We provide the pretrained weights for the backbone and our trained D&A-CDNet models.
| Model | Download Link | Code |
| :--- | :--- | :--- | :--- |
| **Backbone** | [Baidu Netdisk](https://pan.baidu.com/s/1cwkNe2cN6XnMnaZg99bGNQ) | `9810` |
## Usage

### 1. Training
You can train the model using the following command. 

```bash
python train.py --dataset_root ./data --data_name CDD --backbone pvt_v2_b1 --pretrained True --batchsize 8 --epoch 200 --lr 0.0001 --lambda_tc 0.5 --lambda_cs 0.5 --contrastive_margin 1.0 
```

### 2. Evaluation
To evaluate the model on the test set:

```bash
python test.py --dataset_root ./data --data_name CDD --backbone pvt_v2_b1 --checkpoint_path ./checkpoints/best_model.pth(Your pth file)
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
