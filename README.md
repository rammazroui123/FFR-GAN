# cGAN FFR Physiological Plausibility Scoring

## Project Aim
This project develops a conditional Generative Adversarial Network (cGAN) to learn the distribution of physiologically plausible Fractional Flow Reserve (FFR) values from a trusted reference CFD dataset. The final goal is not just to generate FFR values, but to use the trained discriminator as a **physiological plausibility scoring model**.  
For any given **(FFR, anatomy)** pair, the discriminator returns a score between **0 and 1**, where:
- **high score** = physiologically consistent with the learned reference distribution
- **low score** = implausible or out-of-distribution

The biased dataset is **not** used as ground truth during training. It is used only for evaluation.

---

## Dataset Description
The project uses four CSV files:

- `real_FFR_summary.csv`  
  Trusted reference FFR values.

- `fake_FFR_summary.csv`  
  Biased FFR values used for evaluation only.

- `real_lhs_stenosis_design.csv`  
  Trusted anatomy / design features corresponding to the real dataset.

- `fake_lhs_stenosis_design.csv`  
  Anatomy / design features corresponding to the fake dataset.

### Final cleaned dataset sizes
After merging, cleaning, and filtering invalid FFR values:
- **Real/reference**: 1066 samples
- **Fake/biased**: 1200 samples

### Real data split
Only the real dataset is split for learning:
- **Train**: 746
- **Validation**: 160
- **Test**: 160

The fake dataset remains separate for evaluation only.

---

## Pipeline Summary
The notebook follows this pipeline:

1. **Load CSV files**  
   Load real/fake FFR files and their corresponding design files.

2. **Merge matching cases**  
   Create a shared key and inner-join the FFR and design tables so each row contains:
   - one FFR value
   - the matching anatomy / condition features

3. **Clean and engineer features**  
   - convert numeric columns
   - handle eccentricity
   - compute severity fraction
   - compute lesion length
   - create heuristic features such as expected FFR, residual FFR, and interaction terms

4. **Filter invalid FFR values**  
   Keep only values in the range `[0,1]`.

5. **Select final model inputs**  
   Use:
   - `FFR` as the value being judged
   - 9 condition features describing anatomy / lesion characteristics

6. **Scale features**  
   Fit the Min-Max scaler on the **real/reference data only**, then apply the same scaler to fake data.

7. **Split only the real dataset**  
   Create train, validation, and test sets from the trusted reference data.

8. **Build the cGAN**  
   - **Generator**: `(noise + anatomy) -> generated FFR`
   - **Discriminator**: `(FFR + anatomy) -> plausibility score`

9. **Warm up the discriminator**  
   Before full GAN training, train the discriminator on:
   - matched real pairs -> high score
   - shuffled mismatched pairs -> low score

10. **Run adversarial training**  
    Train the full cGAN across 400 epochs.

11. **Use the discriminator as the final model**  
    After training, the discriminator is kept as the final physiological plausibility scoring model.

12. **Evaluate performance**  
    Using:
    - score summaries
    - threshold analysis
    - distribution plots
    - ROC / PR metrics
    - seed-stability testing

---

## Model Summary

### Generator
The generator takes:
- a **16-dimensional noise vector**
- a **9-feature condition vector**

and outputs:
- **one generated FFR value**

Architecture:
- Dense(128) + LeakyReLU + BatchNorm
- Dense(128) + LeakyReLU + BatchNorm
- Dense(64) + LeakyReLU
- Dense(1, sigmoid)

**Generator parameters**
- Total params: **29,185**
- Trainable params: **28,673**
- Non-trainable params: **512**

### Discriminator
The discriminator takes:
- **one FFR value**
- **the 9 anatomical condition features**

and outputs:
- **one plausibility score in [0,1]**

Architecture:
- Dense(256) + LeakyReLU + Dropout
- Dense(128) + LeakyReLU + Dropout
- Dense(64) + LeakyReLU
- Dense(32) + LeakyReLU
- Dense(1, sigmoid)

**Discriminator parameters**
- Total params: **46,081**
- Trainable params: **46,081**
- Non-trainable params: **0**

### Training Strategy
Training is done in two stages:

1. **Discriminator warm-up**
   - uses only trusted real/reference data
   - real matched pairs are positives
   - shuffled mismatched pairs are negatives

2. **Full adversarial training**
   - discriminator is updated twice per generator update
   - generator tries to produce plausible FFR values conditioned on anatomy
   - discriminator learns to score real matched pairs high and generated / mismatched pairs low

---

## Main Results

### Warm-up Performance
After unsupervised discriminator warm-up:

- **D(real)** = **0.835**
- **D(shuffled)** = **0.125**
- **D(fake)** = **0.129**

This showed that even before full adversarial training, the discriminator had already learned to assign high plausibility to correctly matched real pairs and low plausibility to shuffled or biased pairs.

---

### Final Plausibility Score Summary
Final score summary from one representative run:

| Dataset | n | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|---:|
| real_train | 746 | 0.7655 | 0.1229 | 0.3011 | 0.8964 |
| real_val | 160 | 0.7617 | 0.1267 | 0.3725 | 0.8950 |
| real_test | 160 | 0.7665 | 0.1241 | 0.3635 | 0.8963 |
| fake_biased | 1200 | 0.1114 | 0.1687 | 0.0460 | 0.8618 |

### Interpretation
- The real train, validation, and test sets all received **consistently high mean plausibility scores** around **0.76**
- The fake biased dataset received a much lower mean score of **0.1114**
- This indicates **strong separation** between the trusted physiological reference distribution and the biased dataset

---

### Threshold-Based Separation
Thresholds were derived from the **real test score distribution**.

#### 10th percentile threshold
- Threshold = **0.5830**
- Real test below threshold = **10.0%**
- Fake below threshold = **93.8%**

#### 25th percentile threshold
- Threshold = **0.7242**
- Real test below threshold = **25.0%**
- Fake below threshold = **98.9%**

### Interpretation
A threshold that only flags the lowest 10% or 25% of trusted real cases still captures the overwhelming majority of fake cases.

---

### Distribution-Level and Threshold-Based Metrics
Representative evaluation metrics:

#### Distribution-level metrics
- **AUC-ROC** = **0.9853**
- **AUC-PR** = **0.9133**
- **KS statistic** = **0.9125**
- **Mean separation gap** = **0.6551**

#### Threshold-based metrics at the 10th percentile threshold (0.5830)
- **Sensitivity / Recall** = **0.9000**
- **Specificity** = **0.9375**
- **Precision** = **0.6575**
- **F1 score** = **0.7599**
- **False positive rate** = **0.0625**
- **Accuracy** = **0.9331**

#### Confusion matrix at threshold = 0.5830
- **True Positives (real kept)** = **144**
- **False Negatives (real rejected)** = **16**
- **True Negatives (fake rejected)** = **1125**
- **False Positives (fake kept)** = **75**

### Interpretation
- The discriminator achieved **near-perfect ranking ability** with AUC-ROC = **0.9853**
- The score distributions were strongly separated, supported by the high KS statistic
- At the selected threshold, the model correctly classified **93.31%** of all samples

---

### Training Behaviour
Example checkpoints during adversarial training:

- **Epoch 001/400**: D(real_val) = 0.877, D(fake) = 0.160
- **Epoch 100/400**: D(real_val) = 0.872, D(fake) = 0.103
- **Epoch 200/400**: D(real_val) = 0.842, D(fake) = 0.099
- **Epoch 300/400**: D(real_val) = 0.847, D(fake) = 0.096
- **Epoch 400/400**: D(real_val) = 0.762, D(fake) = 0.111

These results show that throughout training, the discriminator consistently assigned higher scores to real validation samples than to fake samples.

---

### Seed Stability / Reproducibility
The full pipeline was repeated across **5 independent random seeds**:
- 42
- 7
- 123
- 99
- 2024

#### Summary across seeds
- **Real mean score** = **0.8281 ± 0.0092**
- **Fake mean score** = **0.1006 ± 0.0023**
- **Separation gap** = **0.7276 ± 0.0103**
- **AUC-ROC** = **0.9973 ± 0.0005**
- **AUC-PR** = **0.9778 ± 0.0106**
- **KS statistic** = **0.9448 ± 0.0074**
- **Fake detection at 10th percentile** = **99.95% ± 0.04%**

### Interpretation
The model was not only strong on a single run, but also **highly stable across multiple random seeds**, indicating strong reproducibility.

---

## How to Run the Notebook

### Requirements
Install:
- Python 3.x
- NumPy
- pandas
- matplotlib
- TensorFlow
- scikit-learn
- SciPy
