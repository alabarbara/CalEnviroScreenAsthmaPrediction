# CalEnviroScreen Asthma Prediction

Research question: Can environmental exposures and socioeconomic characteristics predict a census tract's asthma emergency-department visit rate?

Primary workflow is in Python under `src/`. Run scripts from the repo root (or with paths that resolve to `data/`).

---

## Data flow at a glance

```
asthma_development_unimputed.rds  (development set + fold_id)
        │
        ▼
src/eda.py
  FE (region, haz_waste_any, solid_waste_any)
  → drop haz_waste, solid_waste, education, housing_burden
  → keep covariates unimputed (NOT asthma)
  → keep existing fold_id
        │
        ▼
data/clean_data/eda_training.csv   ← single training CSV (unimputed covariates)
        │
        ├──────────────────────────────┐
        ▼                              ▼
src/validation_baseline.py      src/modelfit.py
  drop NA-asthma rows             drop NA-asthma rows
  → reuse fold_id                 → one-hot categoricals
  → training-mean 5-fold          → per outer fold: train-only median impute
  → pooled CV RMSE                  → A_base / B_poly; linear, lasso, RF
        │
        ├─► asthma_development_data.rds
        ├─► asthma_baseline_by_fold.csv
        └─► asthma_baseline_summary.csv
```

---

## Setup

```bash
pip install pandas numpy matplotlib scipy pyreadr scikit-learn
```

```bash
python src/eda.py
python src/validation_baseline.py
python src/modelfit.py
```

---

## EDA (`src/eda.py`)

**Input:** `data/split_data/asthma_development_unimputed.rds` (fallback: `asthma_development.rds`).

**Exploration:** distributions, correlations, subgroups, missingness on the full pre-drop covariate list.

**Modeling export (order):**

1. Engineered categoricals: `haz_waste_any`, `solid_waste_any`, `region` (Bay Area / San Joaquin Valley / SoCal / Other)
2. Drop: `haz_waste`, `solid_waste`, `education`, `housing_burden`
3. **Do not** impute covariates or `asthma`; keep existing `fold_id` (fold-wise median imputation happens in `modelfit.py`)

**Output:** `data/clean_data/eda_training.csv` only.

(`src/EDA_notebook.Rmd` is the original R notebook; `eda.py` is the maintained equivalent.)

---

## Validation baseline (`src/validation_baseline.py`)

**Input:** `data/clean_data/eda_training.csv`

**Before CV:** coerce `asthma` to numeric; **drop rows with missing `asthma`** (no outcome imputation).

**CV:** keep existing `fold_id` (1–5). For each fold `k`, predict the training-fold mean of `asthma` on fold `k`. Covariates are unused by this baseline.

**Metric:** pooled CV RMSE on all out-of-fold predictions (not the average of fold RMSEs).

**Outputs** (under `data/split_data/`):

| File | Contents |
|------|----------|
| `asthma_development_data.rds` | Eligible development rows used for CV |
| `asthma_baseline_by_fold.csv` | Per-fold training mean, `n`, RMSE |
| `asthma_baseline_summary.csv` | Pooled CV RMSE, `development_n`, `excluded_missing_asthma` |

---

## Model fitting (`src/modelfit.py`)

**Input:** `data/clean_data/eda_training.csv` (same NA-asthma drop and `fold_id` reuse as the baseline).

**Features:** all non-ID covariates; categoricals (`region`, `haz_waste_any`, `solid_waste_any`) one-hot encoded once.

**Per outer fold (shared by all models):** fit numeric medians on the training fold only → apply to train and validation → then build the feature set.

| Feature set | Description |
|-------------|-------------|
| `A_base` | Numeric predictors + one-hot categoricals |
| `B_poly` | `A_base` plus squared mean-centered terms for `poverty`, `unemployment`, `linguistic_isolation`, `ozone`, `pm2_5` (means from the **training fold only**, after imputation) |

**Models** (each × both feature sets, same 5-fold CV):

| Model | Notes |
|-------|--------|
| Linear (OLS) | `StandardScaler` fit on each outer training fold |
| Lasso | `Pipeline(StandardScaler, Lasso)` tuned with `GridSearchCV(cv=5)` so scaling is re-fit inside each inner fold |
| Random forest | 300 trees, `min_samples_leaf=5`, `random_state=42` (unscaled) |

Prints per-fold and pooled OOF RMSE for each model × feature-set combination.

---

## Modeling notes

- Covariate medians are fit on each **outer training fold** only and applied to that fold’s validation covariates; still score only on observed `asthma`.
- For Lasso, standardization is nested inside α tuning (not fit once on the full outer training fold before inner splits).
- Baseline and later models share `eda_training.csv` + the same `fold_id`, after the pre-CV NA-asthma drop.
- Holdout / evaluation data under `data/split_data/` must remain untouched until Part 2.
