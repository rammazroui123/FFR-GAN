# FFR-cGAN: Physiological Coherence Modelling for Fractional Flow Reserve

## Project Overview
This final year dissertation project investigates whether a **conditional Generative Adversarial Network** (cGAN)** can learn the distribution of **physiologically plausible Fractional Flow Reserve (FFR)** values from a reference Computational Fluid Dynamics (CFD) dataset and use that learned distribution to assess biased data.

The work is motivated by the problem that some CFD-generated datasets may contain **physiologically inconsistent FFR values**, particularly when stenosis severity is defined using a biased or non-equivalent formulation. In this project, a cGAN is trained only on the **reference (real) dataset**, which is treated as the ground truth physiological distribution.

The biased dataset is **not used as ground truth** and is **not used for supervised learning**. Instead, it is used only for **evaluation**, in order to determine whether its samples appear out-of-distribution and whether model-based adjustment moves them closer to a physiologically coherent FFR distribution.

---

## Aim
The aim of this project is to build a **conditional GAN (cGAN)** that learns physiologically coherent FFR patterns from a trusted CFD reference dataset and uses this learned representation to:

- identify whether biased samples are out-of-distribution,
- evaluate physiological coherence of generated or adjusted FFR values,
- assess whether biased cases can be shifted toward the learned real-data distribution.

---

## Tools and Libraries
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- scikit-learn  
- Google Colab for training and testing

---

## Research Objective
The central objective is:

> To model the distribution of physiologically plausible FFR values using a conditional GAN trained only on the reference CFD dataset, and to evaluate whether biased CFD samples deviate from this learned physiological distribution.

---

## Method Summary
This project follows a **distribution-learning framework**, not a supervised prediction framework.

### Key methodological principles
- **Model type:** Conditional GAN (cGAN)
- **Training data:** Reference / real CFD dataset only
- **Biased dataset usage:** Evaluation only
- **No supervised learning**
- **No biased labels treated as ground truth**
- **No direct regression target from fake to real**

The model is conditioned on geometry-derived features parsed from case names and metadata. The generator produces candidate FFR values conditioned on these inputs, while the discriminator learns to distinguish real physiological FFR patterns from generated ones.

---

## Dataset Description

### 1. Reference Dataset (Real)
This dataset contains physiologically consistent CFD-derived FFR values and is treated as the **ground truth distribution**.

### 2. Biased Dataset (Fake)
This dataset contains CFD-derived FFR values produced under a biased stenosis formulation. These values may be physiologically inconsistent and are therefore **not treated as ground truth**.

### Important Note
The biased dataset is used **only for evaluation**. It is included to test whether the trained cGAN can detect distributional mismatch and assess whether outputs move closer to the reference physiological distribution.

---

## Input Features
Depending on the experiment version, conditioning features may include parsed geometry information such as:

- stenosis type (e.g. symmetric / asymmetric),
- stenosis severity percentage,
- artery diameter,
- stenosis diameter,
- angle or geometric identifiers,
- case-specific structured features derived from file names.

Example naming patterns include:
- `S40_...` → Symmetric stenosis, 40% severity
- `A50_...` → Asymmetric stenosis, 50–59% severity

The second part of the case name is treated as an incremental identifier and is not used as a physiological feature.

---

## Repository Structure
```text
FFR-cGAN/
│
├── data/
│   ├── real_FFR_summary.csv
│   ├── fake_FFR_summary.csv
│   └── processed/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── cgan_training.ipynb
│   └── evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── parsing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── outputs/
│   ├── figures/
│   ├── training_curves/
│   ├── evaluation_results/
│   └── saved_models/
│
├── README.md
├── requirements.txt
└── .gitignore

University of Birmingham  
GitHub: [@rammazroui123](https://github.com/rammazroui123)
