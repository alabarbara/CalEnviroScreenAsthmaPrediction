import numpy as np
import pandas as pd
import pyreadr
from pathlib import Path

# ============================================================
# 1. Load the existing development data
# ============================================================

# If this relative path fails, use:
# development_data <- readRDS(file.choose())
# Select asthma_development.rds, NOT the holdout file.

_result = pyreadr.read_r("../data/split_data/asthma_development.rds")
development_data = next(iter(_result.values()))

assert "asthma" in list(development_data.columns)
assert "fold_id" in list(development_data.columns)
assert "census_tract" in list(development_data.columns)


# ============================================================
# 2. Prepare the asthma outcome
# ============================================================

# Convert to numeric and identify missing-value markers.
asthma_raw = development_data["asthma"]
asthma_text = pd.Series(
    [
        np.nan if pd.isna(v) else str(v).strip()
        for v in asthma_raw
    ],
    index=asthma_raw.index,
    dtype=object,
)

missing_marker = asthma_text.isna() | asthma_text.str.upper().isin(
    ["", "NA", "N/A"]
)

asthma_numeric = (
    pd.to_numeric(asthma_text, errors="coerce")
    .to_numpy(dtype=float)
    .copy()
)

# Stop if unexpected text could not be converted.
unexpected = (~missing_marker.to_numpy()) & np.isnan(asthma_numeric)

if np.any(unexpected):
    raise RuntimeError("Unexpected nonnumeric values in the asthma column.")

asthma_numeric[missing_marker.to_numpy()] = np.nan

if np.any(~np.isfinite(asthma_numeric[~np.isnan(asthma_numeric)])):
    raise RuntimeError("The asthma outcome contains infinite values.")

# Exclude missing outcomes; do not impute them.
eligible = ~np.isnan(asthma_numeric)

print(
    "Development observations before exclusions:",
    development_data.shape[0],
)

print(
    "Excluded for missing asthma outcome:",
    int((~eligible).sum()),
)

asthma_development = development_data.loc[eligible].copy()

asthma_development["asthma"] = asthma_numeric[eligible]

# Remove the old binary response to avoid using it accidentally.
if "y" in asthma_development.columns:
    asthma_development = asthma_development.drop(columns=["y"])

y_asthma = asthma_development["asthma"].to_numpy(dtype=float)

# Retain the saved fold assignments.
fold_id = asthma_development["fold_id"].to_numpy()

assert not np.any(pd.isna(fold_id))
assert np.all(np.isin(fold_id, np.arange(1, 6)))
assert np.all(np.isin(np.arange(1, 6), fold_id))

print(
    "Eligible development observations:",
    asthma_development.shape[0],
)

print(pd.Series(fold_id).value_counts().sort_index().rename_axis("Fold"))


# ============================================================
# 3. Define RMSE
# ============================================================

def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


# ============================================================
# 4. Evaluate the training-mean baseline
# ============================================================

baseline_prediction = np.full(len(y_asthma), np.nan)

fold_results_asthma = pd.DataFrame(
    {
        "fold": np.arange(1, 6),
        "training_mean": np.nan,
        "validation_n": pd.array([pd.NA] * 5, dtype="Int64"),
        "validation_rmse": np.nan,
    }
)

for k in range(1, 6):

    training = fold_id != k
    validation = fold_id == k

    # Calculate the mean using training observations ONLY.
    mean_k = float(np.mean(y_asthma[training]))

    # Assign that mean to every validation observation.
    baseline_prediction[validation] = mean_k

    fold_results_asthma.loc[k - 1, "training_mean"] = mean_k

    fold_results_asthma.loc[k - 1, "validation_n"] = int(validation.sum())

    fold_results_asthma.loc[k - 1, "validation_rmse"] = rmse(
        y_asthma[validation],
        baseline_prediction[validation],
    )

assert not np.any(np.isnan(baseline_prediction))

# Pool squared errors across all out-of-fold predictions,
# then take the square root.
# Do NOT simply average the five fold RMSE values.
baseline_cv_rmse = rmse(
    y_asthma,
    baseline_prediction,
)


# ============================================================
# 5. Display the result for part (c)
# ============================================================

print(fold_results_asthma)

print(
    "\nAsthma baseline cross-validated RMSE:",
    round(baseline_cv_rmse, 4),
)


# ============================================================
# 6. Save development data and results
# ============================================================

Path("../data/split_data").mkdir(parents=True, exist_ok=True)

# Save separately; preserve the original development file.
pyreadr.write_rds(
    "../data/split_data/asthma_development_data.rds",
    asthma_development.reset_index(drop=True),
)

fold_results_asthma.to_csv(
    "../data/split_data/asthma_baseline_by_fold.csv",
    index=False,
)

pd.DataFrame(
    {
        "model": ["Training-mean baseline"],
        "cv_rmse": [baseline_cv_rmse],
        "development_n": [len(y_asthma)],
    }
).to_csv(
    "../data/split_data/asthma_baseline_summary.csv",
    index=False,
)

print(
    "\nSummary saved to:\n",
    str(
        Path.cwd()
        / ".."
        / "data"
        / "split_data"
        / "asthma_baseline_summary.csv"
    ),
)
