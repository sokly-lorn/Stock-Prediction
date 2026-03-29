# Stock Movement Prediction — Concept Guide

This document explains the theory, mathematics, and methodology behind every model used in `stock_prediction.ipynb`. It is meant to be read alongside the notebook, replacing the inline comments that were removed from the code.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Feature Engineering](#2-feature-engineering)
3. [Data Leakage and Time-Based Splitting](#3-data-leakage-and-time-based-splitting)
4. [Feature Scaling](#4-feature-scaling)
5. [Logistic Regression](#5-logistic-regression)
6. [Random Forest](#6-random-forest)
7. [LSTM](#7-lstm)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Results Summary](#9-results-summary)

---

## 1. Problem Definition

The task is **binary classification**: given the last 30 days of AAPL data, predict whether tomorrow's closing price will be higher than today's.

```
Target y = 1  if Close(t+1) > Close(t)   → "Up"
Target y = 0  otherwise                   → "Down"
```

This is constructed programmatically as:

```python
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
```

`shift(-1)` moves the Close column one row upward, so each row's target refers to the *next* day's price movement.

---

## 2. Feature Engineering

Three categories of features are constructed from the raw Close price series.

### 2.1 Lag Features (30 features)

```python
df[f'lag_{k}'] = df['Close'].shift(k)    for k = 1, 2, ..., 30
```

`lag_k` gives the closing price exactly k days ago. Together, these 30 features give the model a 30-day lookback window of raw price history — its primary source of temporal information.

### 2.2 Daily Return

```
return_1d(t) = [Close(t) - Close(t-1)] / Close(t-1)
```

Implemented via `df['Close'].pct_change()`. This normalises price change into a percentage, making it comparable across different price levels and time periods. It captures day-over-day momentum.

### 2.3 Rolling Mean (5-day)

```
rolling_mean_5(t) = (1/5) * Σ Close(t-k)    for k = 0, 1, 2, 3, 4
```

A simple moving average over the last 5 days. It smooths short-term noise and represents the recent price trend. When the current price is above the rolling mean, it indicates upward momentum.

### 2.4 Rolling Standard Deviation (5-day)

```
rolling_std_5(t) = sqrt[ (1/4) * Σ (Close(t-k) - rolling_mean_5(t))² ]    for k = 0..4
```

Uses Bessel's correction (denominator n−1). Measures short-term price volatility. High values indicate unstable, high-variance price behaviour over the past 5 days.

**Final feature matrix shape:** (N, 33) — 30 lags + return_1d + rolling_mean_5 + rolling_std_5.

---

## 3. Data Leakage and Time-Based Splitting

Financial time series must always be split chronologically. Random shuffling would allow the model to train on future data and test on past data — artificially inflating performance in a way that would never generalise to live trading.

```python
split_idx = int(len(df_model) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

This preserves temporal order: the model trains on the first 80% of dates and is evaluated on the most recent 20% — exactly as it would be deployed in practice.

**The scaler is fitted only on training data**, then applied to the test set using the training statistics. Fitting on test data would constitute leakage because it would incorporate future distributional information into the preprocessing step.

---

## 4. Feature Scaling

### Why it matters for Logistic Regression

Logistic Regression optimises weights via gradient descent. When features have vastly different scales (e.g. `lag_1 ≈ 213` vs `return_1d ≈ 0.003`), the gradient updates are dominated by large-scale features, causing slow or unstable convergence. Scaling ensures each feature contributes proportionally.

### StandardScaler

```
z = (x - μ) / σ
```

Where μ is the feature mean and σ is the standard deviation, both computed from the training set. After transformation, each feature has mean 0 and standard deviation 1.

```python
scaler = StandardScaler()
X_train_lr = scaler.fit_transform(X_train)  # learns μ, σ from train; applies transformation
X_test_lr  = scaler.transform(X_test)       # applies same μ, σ — no refitting
```

### Why Random Forest does not need scaling

Decision trees split on thresholds, not on magnitudes. A split `lag_1 > 215` is equivalent regardless of whether the feature is scaled. Tree-based models are therefore invariant to monotonic transformations of the features.

---

## 5. Logistic Regression

### 5.1 Model

Logistic Regression is a linear classifier. It computes a weighted sum of input features, then passes the result through the **sigmoid function** to produce a probability.

**Step 1 — Linear combination (logit):**

```
z = w₁x₁ + w₂x₂ + ... + w₃₃x₃₃ + b
  = wᵀx + b
```

Where:
- `x₁, ..., x₃₃` are the 33 scaled features for one trading day
- `w₁, ..., w₃₃` are learned weights, one per feature
- `b` is the bias (intercept)

**Step 2 — Sigmoid activation:**

```
P(Up | x) = σ(z) = 1 / (1 + e^{-z})
```

The sigmoid function maps any real number to (0, 1), which is interpreted as the probability of the positive class (Up = 1).

**Step 3 — Decision rule:**

```
ŷ = 1  if P(Up | x) ≥ 0.5
ŷ = 0  otherwise
```

### 5.2 Loss Function

Logistic Regression minimises **Binary Cross-Entropy** (also called Log-Loss):

```
L(w) = -(1/N) * Σᵢ [ yᵢ log(pᵢ) + (1 - yᵢ) log(1 - pᵢ) ]
```

Where:
- `yᵢ ∈ {0, 1}` is the true label for sample i
- `pᵢ = σ(wᵀxᵢ + b)` is the predicted probability
- N is the number of training samples

This loss has no closed-form solution — unlike linear regression's Mean Squared Error which can be solved analytically via `w = (XᵀX)⁻¹Xᵀy`. The sigmoid's non-linearity makes it impossible to isolate w algebraically. The model therefore uses **iterative optimisation** to find the minimum.

### 5.3 Optimiser: L-BFGS

Sklearn's `LogisticRegression` defaults to the **lbfgs** solver (Limited-memory Broyden–Fletcher–Goldfarb–Shanno), not plain gradient descent. Understanding the difference matters.

**Plain gradient descent** uses only the first derivative (gradient) to take small fixed steps downhill:

```
w ← w - α * ∇L(w)
```

Where α is a fixed learning rate. It is simple but slow — it has no information about the curvature of the loss surface, so it takes the same size step regardless of whether the surface is flat or steep.

**L-BFGS** is a second-order quasi-Newton method. It approximates the inverse Hessian (second derivative matrix) to estimate the curvature of the loss surface at the current point, then uses that curvature to take a geometrically informed step:

```
w ← w - H⁻¹ * ∇L(w)
```

Where `H⁻¹` is the approximate inverse Hessian. In regions where the loss surface curves sharply, L-BFGS takes small cautious steps. In flat regions, it takes large steps. This makes it significantly more efficient than gradient descent for small-to-medium datasets — typically converging in far fewer iterations.

The "Limited-memory" prefix means L-BFGS does not store the full N×N Hessian matrix (which would be prohibitively large). Instead it stores only the last m gradient vectors (typically m=10) and reconstructs a low-rank approximation of H⁻¹ on the fly.

`max_iter=500` therefore means up to 500 **L-BFGS iterations**, each of which already incorporates curvature information. In practice, the solver converges well before this limit on a dataset of this size.

### 5.4 Interpretation of Weights

After fitting, `lr.coef_` contains 33 values. A large positive weight for a feature means that feature strongly pushes the prediction toward Up. A large negative weight pushes toward Down. A weight near zero indicates the feature has little predictive value.

---

## 6. Random Forest

### 6.1 Building Block: Decision Tree

A single decision tree recursively partitions the feature space using binary splits. At each node, it selects the feature and threshold that best separates the classes, measured by **Gini Impurity**:

```
Gini(S) = 1 - Σₖ pₖ²
```

Where `pₖ` is the proportion of class k in the node's sample set S. For binary classification:

```
Gini(S) = 1 - (p_up² + p_down²)
```

- **Pure node** (all one class): `Gini = 1 - (1² + 0²) = 0`
- **Maximally impure** (50/50 split): `Gini = 1 - (0.5² + 0.5²) = 0.5`

The best split minimises the **weighted average Gini** of the two child nodes:

```
Gini_split = (|S_left| / |S|) * Gini(S_left) + (|S_right| / |S|) * Gini(S_right)
```

The tree searches all features and all thresholds to find the split minimising this quantity.

With `max_depth=None`, the tree grows until every leaf is pure — perfectly memorising training data. This is intentional, as Random Forest's ensemble mechanism corrects for the resulting overfitting.

### 6.2 Ensemble Mechanism

Random Forest addresses the overfitting of individual trees through two sources of randomness:

**Bagging (Bootstrap Aggregating):**

Each of the 300 trees is trained on a bootstrap sample — N rows drawn with replacement from the training set. On average, each bootstrap sample contains approximately 63.2% of unique training rows (the remaining 36.8%, called "out-of-bag" samples, are never seen by that tree).

**Feature Randomness:**

At every split within a tree, only a random subset of `m` features is considered as candidates. For classification, the default is:

```
m = floor(sqrt(p))   where p = total number of features
m = floor(sqrt(33)) = 5
```

This prevents any single dominant feature from appearing at every split of every tree, forcing the ensemble to discover diverse patterns across all 33 features.

### 6.3 Prediction

For a new test sample x, all 300 trees independently produce a prediction. The final class is determined by majority vote:

```
ŷ = argmax_k Σᵢ 𝟙[tree_i(x) = k]    for k ∈ {0, 1}
```

Equivalently, the predicted probability of class k is the fraction of trees voting for k:

```
P(Up | x) = (1/300) * Σᵢ 𝟙[tree_i(x) = 1]
```

### 6.4 Why It Outperforms a Single Tree

Because each tree is trained on different data and uses different features at each split, their errors are **uncorrelated**. When averaging many uncorrelated, slightly-wrong predictors, random errors cancel out while consistent signal reinforces. This is the bias-variance tradeoff in action: each tree has low bias (fully grown) and high variance (sensitive to its bootstrap sample), but the ensemble has low bias and reduced variance.

### 6.5 Key Parameters

| Parameter | Value | Effect |
|---|---|---|
| `n_estimators` | 300 | Number of trees. More trees reduce variance at the cost of compute. |
| `max_depth` | None | Trees grow until pure. High variance per tree, corrected by ensemble. |
| `random_state` | 42 | Fixes the random seed — ensures identical results across runs. |

---

## 7. LSTM

### 7.1 Motivation

Logistic Regression and Random Forest treat each day's 33 features as an independent, flat vector. They have no native sense of order — the features encode temporal history through the lag columns, but the model itself is not sequence-aware.

LSTM (Long Short-Term Memory) processes data as a **temporal sequence**. It reads 30 consecutive timesteps one at a time, maintaining a hidden state that carries information forward — allowing it to learn patterns that span across time, not just within a single row.

### 7.2 Input Shape

LSTM requires 3-dimensional input: `(samples, timesteps, features)`.

In this notebook:
- `samples` = number of training sequences
- `timesteps` = 30 (the lookback window)
- `features` = 33 (all engineered features)

Each sample is a 30×33 matrix representing 30 consecutive trading days, each described by 33 features.

### 7.3 The LSTM Cell

At each timestep t, the LSTM cell computes four quantities using the current input xₜ and the previous hidden state hₜ₋₁:

**Forget gate** — what to erase from long-term memory:
```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
```

**Input gate** — what new information to write:
```
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)
c̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)
```

**Cell state update** — long-term memory:
```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ c̃ₜ
```

**Output gate** — what to expose as hidden state:
```
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
hₜ = oₜ ⊙ tanh(Cₜ)
```

Where ⊙ is element-wise multiplication, σ is sigmoid, and W, b are learned weight matrices and bias vectors.

After processing all 30 timesteps, the final hidden state h₃₀ is passed to the Dense output layer.

### 7.4 Architecture

```
Input: (batch, 30, 33)
  → LSTM(64)          — 64 memory units, outputs h₃₀ of shape (batch, 64)
  → Dropout(0.3)      — randomly zeroes 30% of units during training
  → Dense(32, relu)   — intermediate representation
  → Dropout(0.2)
  → Dense(1, sigmoid) — outputs P(Up) ∈ (0, 1)
```

**Dropout** prevents the network from memorising training sequences by randomly disabling neurons during each forward pass. This forces the model to learn redundant representations, improving generalisation on unseen data.

### 7.5 Sequence Construction

Sequences are built using a sliding window:

```
X[i] = feature matrix of days [i, i+1, ..., i+29]   shape: (30, 33)
y[i] = Target of day i+30                             scalar: 0 or 1
```

This guarantees X and y are correctly aligned — the 30-day window always predicts the *next* day's movement.

### 7.6 Scaling for LSTM

MinMaxScaler is used instead of StandardScaler because:

```
x_scaled = (x - x_min) / (x_max - x_min)   → output ∈ [0, 1]
```

Neural network activations (sigmoid, tanh) saturate for large input magnitudes, causing vanishing gradients. Compressing inputs to [0, 1] keeps activations in their sensitive, non-saturating region during early training.

As with the baseline models, the scaler is fitted only on training sequences and applied to test sequences using training statistics.

### 7.7 Training

- **Loss:** Binary Cross-Entropy (same as Logistic Regression)
- **Optimiser:** Adam — an adaptive gradient descent variant that maintains per-parameter learning rates
- **EarlyStopping:** monitors `val_loss` with patience=10. Stops training and restores the best weights when validation loss stops improving for 10 consecutive epochs, preventing overfitting.

---

## 8. Evaluation Metrics

All three models are evaluated on the same held-out test window using four metrics derived from the **confusion matrix**.

### 8.1 Confusion Matrix

|  | Predicted Down (0) | Predicted Up (1) |
|---|---|---|
| **Actually Down (0)** | True Negative (TN) | False Positive (FP) |
| **Actually Up (1)** | False Negative (FN) | True Positive (TP) |

### 8.2 Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

The proportion of all predictions that were correct. Can be misleading if classes are imbalanced.

### 8.3 Precision

```
Precision = TP / (TP + FP)
```

Of all days predicted as Up, what fraction actually went Up. High precision means few false alarms — the model is conservative but reliable when it predicts Up.

### 8.4 Recall

```
Recall = TP / (TP + FN)
```

Of all days that actually went Up, what fraction did the model correctly identify. High recall means the model catches most Up days but may produce more false positives.

### 8.5 F1-Score

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

The harmonic mean of Precision and Recall. Useful as a single summary metric when both matter equally. It penalises extreme imbalance between the two — a model with Precision=1.0, Recall=0.01 gets F1≈0.02, not 0.5.

---

## 9. Results Summary

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Logistic Regression | 64.9% | 68.3% | 58.3% |
| Random Forest | 66.0% | 69.0% | 60.4% |
| LSTM | 47.7% | 0.0% | 0.0% |

**Logistic Regression** provides a clean, interpretable baseline. Its precision is competitive, but its recall indicates it misses roughly 4 in 10 real Up days. The linear decision boundary limits its ability to capture non-linear price dynamics.

**Random Forest** marginally outperforms LR across all metrics. The ensemble of 300 diverse trees captures non-linear feature interactions that a single linear model cannot, at the cost of interpretability.

**LSTM** is theoretically the most expressive model — it can learn patterns across the full 30-day sequence rather than treating each day's features independently. In practice, performance is sensitive to architecture and data volume. The fixed implementation uses all 33 features, proper sequence alignment, Dropout regularisation, and EarlyStopping on validation loss.
