"""
Training-mean baseline on EDA modeling CSV
============================================================
Python translation of validation_baseline.R — same logic and paths.

Expects data/clean_data/eda_training.csv from EDA_notebook.Rmd / eda.py:
  engineered features, correlated cols dropped, covariates median-imputed,
  existing fold_id retained, asthma NOT imputed.
Drops NA-asthma rows BEFORE CV, then reuses fold_id for 5-fold RMSE.
============================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

# ------------------------------------------------------------
# 1. Load EDA training CSV
# ------------------------------------------------------------

src_dir = Path(__file__).resolve().parent
cwd = Path.cwd()

csv_candidates = [
    src_dir / "../data/clean_data/eda_training.csv",
    cwd / "data/clean_data/eda_training.csv",
]
csv_path = next((p.resolve() for p in csv_candidates if p.resolve().exists()), None)
if csv_path is None:
    raise FileNotFoundError(
        "Could not find eda_training.csv. Knit/run EDA_notebook.Rmd first."
    )

development_data = pd.read_csv(csv_path)

assert "asthma" in development_data.columns
assert "fold_id" in development_data.columns
assert "census_tract" in development_data.columns

print(f"Loaded: {csv_path.resolve()}")


# ------------------------------------------------------------
# 2. Prepare asthma; drop missing outcomes BEFORE CV
# ------------------------------------------------------------

# Match R: trimws(as.character(...)); pandas StringDtype preserves NA
asthma_text = development_data["asthma"].astype("string").str.strip()

missing_marker = (
    development_data["asthma"].isna()
    | asthma_text.isna()
    | asthma_text.str.upper().isin(["", "NA", "N/A"])
)

asthma_numeric = pd.to_numeric(asthma_text, errors="coerce")
asthma_numeric = asthma_numeric.mask(missing_marker)

unexpected = (~missing_marker) & asthma_numeric.isna()
if bool(unexpected.any()):
    raise ValueError("Unexpected nonnumeric values in the asthma column.")

observed = asthma_numeric.dropna()
if len(observed) and not bool(np.isfinite(observed.to_numpy(dtype=float)).all()):
    raise ValueError("The asthma outcome contains infinite values.")

# Exclude missing outcomes before any fold-based scoring.
# Do not impute asthma; covariates were already imputed in EDA.
eligible = asthma_numeric.notna()

print(
    "Development observations before exclusions:",
    len(development_data),
)
print(
    "Excluded for missing asthma outcome:",
    int((~eligible).sum()),
)

asthma_development = development_data.loc[eligible].copy()
asthma_development["asthma"] = asthma_numeric.loc[eligible].astype(float)

# Remove legacy binary response if present (R: asthma_development$y <- NULL)
if "y" in asthma_development.columns:
    asthma_development = asthma_development.drop(columns=["y"])

y_asthma = asthma_development["asthma"].to_numpy(dtype=float)
fold_id = asthma_development["fold_id"].to_numpy()
fold_id_int = fold_id.astype(int)

# Match R: !anyNA(y); !anyNA(fold_id); all(fold_id %in% 1:5); all(1:5 %in% fold_id)
assert not np.isnan(y_asthma).any()
assert not pd.isna(fold_id).any()
assert set(fold_id_int).issubset({1, 2, 3, 4, 5})
assert {1, 2, 3, 4, 5}.issubset(set(fold_id_int))

print(
    "Eligible development observations (post NA-asthma drop):",
    len(asthma_development),
)
print(pd.Series(fold_id_int).value_counts().sort_index().rename_axis("Fold"))


# ------------------------------------------------------------
# 3. RMSE helper
# ------------------------------------------------------------

def rmse(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# ------------------------------------------------------------
# 4. Training-mean baseline (each fold as validation once)
# ------------------------------------------------------------
# Covariates are unused by this baseline. Val scoring uses only
# observed asthma (already guaranteed by the pre-CV drop).

baseline_prediction = np.full(len(y_asthma), np.nan)

fold_results_asthma = pd.DataFrame(
    {
        "fold": list(range(1, 6)),
        "training_mean": np.full(5, np.nan),
        "validation_n": np.full(5, 0, dtype=int),
        "validation_rmse": np.full(5, np.nan),
    }
)

for k in range(1, 6):
    training = fold_id_int != k
    validation = fold_id_int == k

    # Fit on training outcomes only (all observed after pre-CV drop).
    mean_k = float(y_asthma[training].mean())

    baseline_prediction[validation] = mean_k
    fold_results_asthma.loc[fold_results_asthma["fold"] == k, "training_mean"] = mean_k
    fold_results_asthma.loc[fold_results_asthma["fold"] == k, "validation_n"] = int(
        validation.sum()
    )
    fold_results_asthma.loc[fold_results_asthma["fold"] == k, "validation_rmse"] = rmse(
        y_asthma[validation],
        baseline_prediction[validation],
    )

assert not np.isnan(baseline_prediction).any()

# Pooled CV RMSE over all out-of-fold predictions (not mean of fold RMSEs).
baseline_cv_rmse = rmse(y_asthma, baseline_prediction)


# ------------------------------------------------------------
# 5. Display
# ------------------------------------------------------------

print(fold_results_asthma.to_string(index=False))
print(
    "\nAsthma baseline cross-validated RMSE:",
    round(baseline_cv_rmse, 4),
)


# ------------------------------------------------------------
# 6. Save results
# ------------------------------------------------------------

if (src_dir / "../data/split_data").resolve().exists():
    split_dir = (src_dir / "../data/split_data").resolve()
elif (cwd / "data/split_data").resolve().exists():
    split_dir = (cwd / "data/split_data").resolve()
else:
    split_dir = (src_dir / "../data/split_data").resolve()
    split_dir.mkdir(parents=True, exist_ok=True)

# Eligible development rows used for CV (NA asthma already removed).
rds_path = split_dir / "asthma_development_data.rds"
pyreadr.write_rds(str(rds_path), asthma_development)

fold_results_asthma.to_csv(
    split_dir / "asthma_baseline_by_fold.csv", index=False
)

pd.DataFrame(
    [
        {
            "model": "Training-mean baseline",
            "cv_rmse": baseline_cv_rmse,
            "development_n": len(y_asthma),
            "excluded_missing_asthma": int((~eligible).sum()),
        }
    ]
).to_csv(split_dir / "asthma_baseline_summary.csv", index=False)

print(
    "\nSummary saved to:\n",
    (split_dir / "asthma_baseline_summary.csv").resolve(),
)
