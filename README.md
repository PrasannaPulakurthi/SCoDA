# Bridging Domain Shifts Through Self-contrastive Learning And Distribution Alignment  
**IEEE ICIP 2025**  
Repository: SCoDA (SVHN → MNIST)

**Manuscript:** https://ieeexplore.ieee.org/document/11084279

---

## 1. Abstract

This repository implements a demonstration of **SCoDA** framework, a method that bridges domain shifts by combining **self-contrastive learning** (contrastive regularization on two views of each sample) with **distribution alignment** via **Maximum Mean Discrepancy (MMD)**. We instantiate the approach on a standard digits domain adaptation benchmark **SVHN → MNIST**.

---

## 2. Method Overview

Given labeled source images \( \mathcal{D}_S \) and unlabeled target images \( \mathcal{D}_T \), we optimize:

\[
\mathcal{L} = \mathcal{L}_{CE}(\text{source}) \;+\; \lambda \big( \mathcal{L}_{MMD}(f_S, f_T) \;+\; \mathcal{L}_{InfoNCE}(f_S^{(1)}, f_S^{(2)}) \;+\; \mathcal{L}_{InfoNCE}(f_T^{(1)}, f_T^{(2)}) \big).
\]

- **Cross-entropy** on source labels  
- **MK-MMD** between source and target bottleneck features for distribution alignment  
- **Self-contrastive (InfoNCE)** on both domains, using two stochastic views per sample to encourage invariances  

We use a **ResNet-50** pre-trained on ImageNet as the backbone and a small MLP bottleneck, then a linear classifier.

---

## 3. Installation

```bash
conda create -n scoda python=3.10 -y
conda activate scoda
pip install -r requirements.txt
```

---

## 4. Datasets

- SVHN (source) and MNIST (target) are downloaded automatically by torchvision on first run into ./data/.

- We use paired augmentations (two views) for both domains:

    - Random resized crops and color jitter for the training view

    - A deterministic “test” view for validation/contrastive pairing

- Images are converted to 3 channels for MNIST to match the ResNet-50 input.

---

## 5. Acknowledgements

We gratefully acknowledge:

* **Transfer-Learning-Library (tllib)** for MK-MMD and several utilities that inspired and informed parts of this code and design: [https://github.com/thuml/Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)
* **SimCLR (PyTorch implementation)** for contrastive learning inspiration and practical tooling: [https://github.com/sthalles/SimCLR](https://github.com/sthalles/SimCLR)

---

### 6. Citation

If you use this repository, please cite the ICIP 2025 manuscript:

```bash
@inproceedings{scoda_iclip2025,
  title     = {Bridging Domain Shifts Through Self-contrastive Learning and Distribution Alignment},
  booktitle = {IEEE International Conference on Image Processing (ICIP)},
  year      = {2025},
  url       = {https://ieeexplore.ieee.org/document/11084279}
}

```
