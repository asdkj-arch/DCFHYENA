# DCFHYENA

**DCFHYENA: A Dual-Channel Hyena Framework with Bidirectional Interaction Fusion for Transcription Factor Binding Site Prediction**

DCFHYENA is a PyTorch-based deep learning framework for transcription factor binding site (TFBS) prediction. It employs a dual-channel Hyena architecture with Bidirectional Interaction Fusion (BIF) to learn complementary DNA sequence representations and integrate multi-scale features for accurate TFBS prediction.

## Highlights

- 🧬 **Dual-channel architecture:** Learns complementary DNA sequence representations using character-level and overlapping 3-mer tokenization strategies.
- 🔗 **Bidirectional Interaction Fusion:** Integrates complementary features through bidirectional cross-attention and gated fusion.
- 🚀 **Competitive prediction performance:** Achieves strong performance across 165 independent ENCODE TFBS prediction datasets.
- ⚡ **Efficient sequence modeling:** Leverages the Hyena architecture for efficient DNA sequence representation learning.
- 🔬 **Reproducibility:** Provides pretrained and fine-tuned model checkpoints, complete prediction scripts, and instructions for independent TFBS prediction.

---

## Table of Contents

- [1. Repository Structure](#1-repository-structure)
- [2. Requirements](#2-requirements)
- [3. Installation](#3-installation)
- [4. Model Weights](#4-model-weights)
- [5. Data Preprocessing](#5-data-preprocessing)
- [6. Model Loading](#6-model-loading)
- [7. Prediction on Independent TFBS Datasets](#7-prediction-on-independent-tfbs-datasets)
- [8. Prediction Output](#8-prediction-output)
- [9. Reproducing the Reported Results](#9-reproducing-the-reported-results)
- [10. Citation](#10-citation)
- [11. Contact](#11-contact)

---

## 1. Repository Structure

The repository is organized as follows:

```text
DCFHYENA/
│
├── configs/
├── evals/
├── src/
├── ckpt/
├── data/
├── requirements.txt
├── train.py
├── run.bash
├── test.bash
└── LICENSE
```



---

## 2. Requirements

DCFHYENA was implemented using Python and PyTorch.

The recommended software environment is listed below:

| Component | Version |
|---|---|
| Python | 3.8.20 |
| PyTorch | 2.0.0+cu118 |
| CUDA | 11.8 |
| NumPy | 1.24.4 |
| Pandas | 2.0.3 |
| scikit-learn | 1.3.2 |
| Biopython | 1.79 |

GPU acceleration is recommended for efficient model inference.

---

## 3. Installation

### 3.1 Create a Conda environment

```bash
conda create -n dcf python=[3.8] -y
conda activate dcf
```

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively, if an environment configuration file is provided:

```bash
conda env create -f environment.yml
conda activate dcfhyena
```

### 3.3 Verify the installation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## 4. Model Weights

To facilitate reproducibility and independent applications, we provide both pretrained and fine-tuned model checkpoints.

```text
DCFHYENA/
├── ckpt/
│   ├── pretrained
│   └── finetuned
├── data/
│   ├── benchmark/
│   │   ├── HG19
│   │   └── HG38

│   └── genome/
│   │   ├── HG19
│   │   └── HG38
└── ...

```

### 4.1 Pretrained model

The pretrained checkpoint contains the model parameters obtained during genome-scale DNA sequence pretraining.

It can be used to initialize DCFHYENA for downstream TFBS prediction tasks or for further fine-tuning.

### 4.2 Fine-tuned model

The fine-tuned checkpoint contains the parameters optimized for TFBS classification.

This checkpoint is recommended for users who want to directly perform TFBS prediction without repeating the training process.

```

**Important:** The configuration file associated with each checkpoint should be used when loading the model to ensure architectural consistency.

---

## 5. Fine-tuned

```bash
bash run.bash
```

