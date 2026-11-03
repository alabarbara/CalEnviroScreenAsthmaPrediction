# CalEnviroScreen Asthma Prediction

Research question: Can environmental exposures and socioeconomic characteristics predict a census tract's asthma emergency-department visit rate?

---

## Data flow at a glance

```
asthma_development_unimputed.rds  (development set + fold_id)
        │
        ▼
EDA_notebook.Rmd
  FE (region, haz_waste_any, solid_waste_any)
  → drop haz_waste, solid_waste, education, housing_burden
  → global median impute covariates (NOT asthma)
  → keep existing fold_id
        │
        ▼
data/clean_data/eda_training.csv   ← single training CSV
        │
        ▼
validation_baseline.R
  drop NA-asthma rows BEFORE CV
  → reuse fold_id
  → training-mean 5-fold baseline
  → pooled CV RMSE
        │
        ├─► asthma_development_data.rds
        ├─► asthma_baseline_by_fold.csv
        └─► asthma_baseline_summary.csv
```

---

## EDA (`src/EDA_notebook.Rmd`)

**Input:** `data/split_data/asthma_development_unimputed.rds` (fallback: `asthma_development.rds`).

**Exploration:** unchanged (distributions, correlations, subgroups, missingness) using the full pre-drop covariate list.

**Modeling export (order):**

1. Engineered categoricals: `haz_waste_any`, `solid_waste_any`, `region`
2. Drop: `haz_waste`, `solid_waste`, `education`, `housing_burden`
3. Median-impute covariates on the full development set: `pm2_5`, `drinking_water`, `traffic`, `linguistic_isolation`, `poverty`, `unemployment`
4. **Do not** impute `asthma`; keep existing `fold_id`

**Output:** `data/clean_data/eda_training.csv` only.

---

## Validation baseline (`src/validation_baseline.R`)

**Input:** `data/clean_data/eda_training.csv`

**Before CV:** coerce `asthma` to numeric; **drop rows with missing `asthma`** (no outcome imputation).

**CV:** keep existing `fold_id` (1–5). For each fold `k`, predict training-fold mean of `asthma` on fold `k`. Covariates unused by this baseline.

**Metric:** pooled CV RMSE on all out-of-fold predictions (not the average of fold RMSEs).

**Outputs:** `asthma_development_data.rds`, `asthma_baseline_by_fold.csv`, `asthma_baseline_summary.csv` (includes `excluded_missing_asthma`).

---

## Modeling notes

- EDA covariate medians are **global** (fit on all development rows). For stricter no-leakage CV on future models, re-fit medians on each training fold and apply to that fold’s validation covariates; still score only on observed `asthma`.
- Baseline and later models should share `eda_training.csv` + the same `fold_id`, after the pre-CV NA-asthma drop.
