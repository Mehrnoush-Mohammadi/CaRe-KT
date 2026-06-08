# CaRe-KT: Retrieval-Augmented Reasoning for Knowledge Tracing

This repository contains the implementation of **CaRe-KT**, a cache-guided retrieval framework that reframes Knowledge Tracing as relevance-guided reasoning over historical learning interactions.

> Submitted to CIKM 2026 (under review)

---

## Overview

CaRe-KT introduces an **Evidence-Graph Cache (EGC)** that maintains dynamic semantic, conceptual, and temporal relationships across the question space. For each incoming query, EGC constructs a query-specific evidence subgraph that simultaneously anchors retrieval of temporally grounded learning trajectories and governs their fusion via graph attention — unifying evidence retrieval and reasoning through a shared relevance structure.

### Key Components

- **Query-Adaptive Pre-Retrieval Filtering** — two-stage ε-neighborhood strategy over the EGC
- **Context-Aware Retrieval** — temporal anchoring over global interaction memory
- **Behavior-Aware Encoding** — GRU encoder with trajectory-aware contrastive regularization
- **Retrieval-Aware Fusion** — graph attention reusing cached EGC structure for prediction

---

## Requirements

```
Python >= 3.8
PyTorch >= 1.12
transformers
EduData
scikit-learn
numpy
pandas
tqdm
```

Install dependencies:
```bash
pip install torch transformers EduData scikit-learn numpy pandas tqdm
```

---

## Datasets

The following datasets are used and downloaded automatically via `EduData`:

| Dataset | Students | Questions | Concepts | Interactions |
|---|---|---|---|---|
| ASSISTment09 | 4,151 | 16,891 | 110 | 325,673 |
| Slepemapy.cz | 81,700 | 2,900 | 1,458 | 9,786,500 |
| Junyi | 175,400 | 700 | 40 | 25,670,200 |

---

## Usage

Open and run the Jupyter notebook for each dataset:

```bash
jupyter notebook CaReASSISTment09.ipynb
```

Interactions are chronologically split 80/10/10 for training, validation, and testing. Hyperparameters are selected on the validation set.

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Decay coefficient λ | ln(2)/80 |
| Adaptive threshold η | 0.75 |
| Sliding window W | 50 |
| Temperature ζ | 0.1 |
| λ_SCL, λ_self | 1.0 |

---

## Results

CaRe-KT consistently outperforms ten strong KT baselines across all datasets, with particular robustness under sparse-history and cold-start conditions.

| Model | ASSISTment09 AUC | Slepemapy AUC | Junyi AUC |
|---|---|---|---|
| stableKT | 81.91 | 77.53 | 80.56 |
| UKT | 74.78 | 77.96 | 80.21 |
| **CaRe-KT** | **83.05** | **80.15** | **82.79** |

---

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{carekt2026,
  title     = {Retrieval-Augmented Reasoning for Knowledge Tracing},
  booktitle = {Proceedings of the ACM International Conference on
               Information and Knowledge Management (CIKM)},
  year      = {2026}
}
```
