# Predicting Student Health Risk — Kaggle Playground Series S6E7

A multiclass classification solution for Kaggle's [Playground Series - Season 6, Episode 7](https://www.kaggle.com/competitions/playground-series-s6e7), predicting student health status (`at-risk`, `unhealthy`, `fit`) from lifestyle and biometric features.

## Problem

- **Task:** 3-class classification (`health_condition`: at-risk / unhealthy / fit)
- **Evaluation metric:** [Balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) — the average of per-class recall, which matters a lot here because of severe class imbalance
- **Data:** ~690K training rows, ~296K test rows, 13 features (7 numeric biometric/lifestyle measures, 6 categorical lifestyle features), with missingness across most numeric columns

**Class distribution:**
| Class | Share of training data |
|---|---|
| at-risk | 85.9% |
| unhealthy | 8.4% |
| fit | 5.8% |

## Approach

### 1. Baseline: plain LightGBM
A first pass using LightGBM's native categorical and missing-value handling (no encoding or imputation needed) with plain accuracy as the target scored well on accuracy (~96.7% OOF) but poorly on the competition's actual metric — **0.876 balanced accuracy** on the leaderboard. The model was essentially ignoring the minority classes, since accuracy is dominated by the 86%-majority `at-risk` class.

### 2. Class-weighted training
Switched to `sklearn.utils.class_weight.compute_sample_weight('balanced', ...)` to weight the loss function inversely to class frequency, forcing the model to pay attention to `fit` and `unhealthy` despite their small share of the data.

This alone moved the leaderboard score from **0.876 → 0.949**.

### 3. Model blending
Trained a second model type, CatBoost (with `auto_class_weights='Balanced'` and native categorical support), and averaged its predicted probabilities with LightGBM's. Blending gave a small additional lift to **0.9499** balanced accuracy.

### 4. Post-hoc threshold tuning
Directly optimized per-class probability multipliers against the OOF balanced-accuracy metric using `scipy.optimize.minimize` (Nelder-Mead, since balanced accuracy is a non-differentiable step function). This gave essentially no further improvement over the blend — an indication the blended probabilities were already close to their local optimum for this metric.

### 5. Feature engineering (exploratory)
Tried adding BMI category buckets, several ratio features (e.g. steps per exercise minute, calories per BMI point), and per-column missingness indicators. This did **not** improve results — gradient-boosted trees already approximate these relationships internally through their split structure, and the extra features didn't provide meaningful new signal.

## Results

| Approach | Balanced accuracy (leaderboard) |
|---|---|
| Plain LightGBM (unweighted) | 0.876 |
| Class-weighted LightGBM | 0.949 |
| LightGBM + CatBoost blend | **0.9499** (best) |
| Blend + threshold tuning | 0.9498 |
| Blend + feature engineering | 0.9496 |

## Key takeaways

- **Know your metric before you optimize.** The single biggest jump in this project (0.876 → 0.949) came from realizing the leaderboard metric was balanced accuracy, not plain accuracy, and weighting the training loss accordingly — not from any model architecture change.
- **Diminishing returns set in fast on tabular boosted-tree data.** Model blending, threshold tuning, and feature engineering each moved the score by well under a percentage point after the class-weighting fix, versus a ~7-point jump from that one change.
- **A second tree-based model isn't always diverse enough.** CatBoost and LightGBM, both gradient-boosted trees on the same features, made fairly correlated errors — the blend lift was modest. A genuinely different model family (e.g. a neural net, or a linear model on engineered features) would likely decorrelate errors more effectively.

## Tech stack

- `pandas`, `numpy` — data handling
- `lightgbm`, `catboost` — gradient boosting models
- `scikit-learn` — cross-validation, metrics, class weighting
- `scipy.optimize` — threshold tuning

## Reproducing

The full notebook (`student-health-risk-prediction.ipynb`) trains both models with 5-fold stratified cross-validation, blends their out-of-fold and test predictions, and writes submission files at each stage:
- `submission.csv` — LightGBM only, class-weighted
- `submission_blend.csv` — LightGBM + CatBoost blend
- `submission_tuned.csv` — blend with tuned per-class thresholds

Data is expected under `/kaggle/input/competitions/playground-series-s6e7/` (Kaggle notebook environment) or can be pointed at a local path.
