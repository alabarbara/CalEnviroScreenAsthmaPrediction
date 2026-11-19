"""Development-only case bootstrap for the Part 2 Q3a OLS coefficients."""

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

N_BOOT = 2000
ALPHA = 0.05
RANDOM_STATE = 42


def tract_id(series: pd.Series) -> pd.Series:
    """Match part2_inference.R: strip whitespace and trailing .0 from tract IDs."""
    cleaned = series.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
    return cleaned


def load_development_analysis() -> pd.DataFrame:
    """Rebuild the Q3a analysis frame: raw covariates + development region."""
    dev = pd.read_csv(DEV_PATH, dtype={"census_tract": str})
    raw = pd.read_csv(RAW_PATH, dtype={"census_tract": str})

    dev_ids = tract_id(dev["census_tract"])
    raw_ids = tract_id(raw["census_tract"])

    if len(dev) != 6428 or dev_ids.duplicated().any() or raw_ids.duplicated().any():
        raise ValueError("Inputs do not match the expected Part 1 split.")

    # Map each development ID to its raw row position.
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
    return analysis.reset_index(drop=True)


def complete_cases(data: pd.DataFrame) -> pd.DataFrame:
    cols = VARIABLES + ["region"]
    return data.loc[data[cols].notna().all(axis=1), cols].reset_index(drop=True)


def fit_ols(data: pd.DataFrame):
    return smf.ols(FORMULA, data=data).fit()


def bootstrap_p_value(samples: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: coefficient = 0."""
    n = len(samples)
    if n == 0:
        return np.nan
    p_le = np.mean(samples <= 0.0)
    p_ge = np.mean(samples >= 0.0)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def main() -> None:
    analysis = load_development_analysis()
    base = complete_cases(analysis)
    n = len(base)
    if n == 0:
        raise ValueError("No complete cases available for the OLS specification.")

    point = fit_ols(base)
    terms = list(point.params.index)

    rng = np.random.default_rng(RANDOM_STATE)
    boot = np.full((N_BOOT, len(terms)), np.nan, dtype=float)
    n_failed = 0

    for b in range(N_BOOT):
        draw = base.iloc[rng.integers(0, n, size=n)].reset_index(drop=True)
        try:
            fit = fit_ols(draw)
            boot[b, :] = fit.params.reindex(terms).to_numpy(dtype=float)
        except Exception:
            n_failed += 1

    valid = boot[~np.isnan(boot).any(axis=1)]
    if len(valid) < max(100, N_BOOT // 2):
        raise RuntimeError(
            f"Too many failed bootstrap fits ({n_failed} of {N_BOOT})."
        )

    lo = ALPHA / 2.0
    hi = 1.0 - ALPHA / 2.0
    rows = []
    for j, term in enumerate(terms):
        samples = valid[:, j]
        ci_lower, ci_upper = np.quantile(samples, [lo, hi])
        p_value = bootstrap_p_value(samples)
        rows.append(
            {
                "term": term,
                "estimate": float(point.params[term]),
                "bootstrap_mean": float(np.mean(samples)),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "p_value": p_value,
                "significant_at_5_pct": p_value < ALPHA,
                "n_bootstrap": int(len(valid)),
                "development_n_complete": int(n),
            }
        )

    results = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / "bootstrap_estimates.csv"
    results.to_csv(out_path, index=False)

    print(
        f"Development complete cases: {n} of {len(analysis)}; "
        f"successful replicates: {len(valid)} of {N_BOOT}"
    )
    print(results.to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
