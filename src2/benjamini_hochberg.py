"""
Benjamini-Hochberg FDR screen for Part 2 Q3a OLS coefficients.

1. Collect all p-values from the development OLS (including intercept)
2. Sort p-values in ascending order
3. Assign ranks (1 for smallest p-value up to m)
4. Choose alpha = 0.05
5. For each p-value, threshold = alpha * rank / m
6. Flag significant if p-value <= threshold
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "part2"

DEV_PATH = ROOT / "data" / "clean_data" / "eda_training.csv"
RAW_PATH = ROOT / "data" / "calenviroscreen.csv"

VARIABLES = [
    "asthma",
    "poverty",
    "unemployment",
    "linguistic_isolation",
    "ozone",
    "pm2_5",
]
REGION_LEVELS = ["Bay Area", "Other", "San Joaquin Valley", "SoCal"]
FORMULA = (
    "asthma ~ poverty + unemployment + linguistic_isolation + "
    "ozone + pm2_5 + C(region)"
)
ALPHA = 0.05


def tract_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def load_development_complete_cases() -> pd.DataFrame:
    """Same Q3a analysis frame as part2_inference.R / bootstrap.py."""
    dev = pd.read_csv(DEV_PATH, dtype={"census_tract": str})
    raw = pd.read_csv(RAW_PATH, dtype={"census_tract": str})

    dev_ids = tract_id(dev["census_tract"])
    raw_ids = tract_id(raw["census_tract"])

    if len(dev) != 6428 or dev_ids.duplicated().any() or raw_ids.duplicated().any():
        raise ValueError("Inputs do not match the expected Part 1 split.")

    raw_pos = pd.Series(np.arange(len(raw_ids)), index=raw_ids)
    try:
        matched = raw_pos.loc[dev_ids].to_numpy()
    except KeyError as exc:
        raise ValueError("Development tracts missing from raw data.") from exc

    analysis = raw.iloc[matched][VARIABLES].copy()
    analysis["region"] = pd.Categorical(
        dev["region"].to_numpy(), categories=REGION_LEVELS
    )
    if analysis["region"].isna().any():
        raise ValueError("Development region is missing for some tracts.")

    cols = VARIABLES + ["region"]
    return analysis.loc[analysis[cols].notna().all(axis=1), cols].reset_index(
        drop=True
    )


def benjamini_hochberg(p_values: pd.Series, alpha: float = ALPHA) -> pd.DataFrame:
    """Per-rank BH thresholds; significant if p <= alpha * rank / m."""
    m = len(p_values)
    if m == 0:
        raise ValueError("No p-values to adjust.")

    ordered = p_values.sort_values(ascending=True)
    ranks = np.arange(1, m + 1)
    thresholds = alpha * ranks / m
    passes = ordered.to_numpy(dtype=float) <= thresholds

    return pd.DataFrame(
        {
            "term": ordered.index,
            "p_value": ordered.to_numpy(dtype=float),
            "rank": ranks,
            "bh_threshold": thresholds,
            "p_le_threshold": passes,
        }
    )


def main() -> None:
    data = load_development_complete_cases()
    fit = smf.ols(FORMULA, data=data).fit()

    # All coefficients, including the intercept (m = number of terms).
    p_values = fit.pvalues
    bh = benjamini_hochberg(p_values, alpha=ALPHA)
    bh.insert(1, "estimate", fit.params.reindex(bh["term"]).to_numpy(dtype=float))
    bh["alpha"] = ALPHA
    bh["m_tests"] = len(p_values)
    bh["development_n_complete"] = int(fit.nobs)

    # Stable column order for the CSV.
    results = bh[
        [
            "term",
            "estimate",
            "p_value",
            "rank",
            "bh_threshold",
            "p_le_threshold",
            "alpha",
            "m_tests",
            "development_n_complete",
        ]
    ]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / "benjamini_hochberg.csv"
    results.to_csv(out_path, index=False)

    print(
        f"Development complete cases: {int(fit.nobs)}; "
        f"tests (incl. intercept): {len(p_values)}; alpha={ALPHA}"
    )
    print(results.to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
