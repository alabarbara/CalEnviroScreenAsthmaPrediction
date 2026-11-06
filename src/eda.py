"""
Exploratory Data Analysis — Python translation of EDA_notebook.Rmd.
Maintains the same logic, feature engineering, analyses, and modeling export.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive; figures closed after each plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
from scipy import stats

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")


def _show():
    """Stand-in for Rmd inline plots: render then close (Agg backend)."""
    plt.tight_layout()
    plt.close("all")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
# Prefer repo-relative paths (same candidate order as the Rmd)
src_dir = Path(__file__).resolve().parent
cwd = Path.cwd()

# Match Rmd relative candidates: ../data/... then data/...
dev_candidates = [
    src_dir / "../data/split_data/asthma_development_unimputed.rds",
    cwd / "data/split_data/asthma_development_unimputed.rds",
    src_dir / "../data/split_data/asthma_development.rds",
    cwd / "data/split_data/asthma_development.rds",
]
dev_path = next((p.resolve() for p in dev_candidates if p.resolve().exists()), None)
if dev_path is None:
    raise FileNotFoundError("Could not find asthma development RDS under data/split_data/.")

result = pyreadr.read_r(str(dev_path))
dat = next(iter(result.values())).copy()
assert "fold_id" in dat.columns and "asthma" in dat.columns
print(f"Loaded: {dev_path.resolve()}")

# ---------------------------------------------------------------------------
# Set which columns are which
# ---------------------------------------------------------------------------
outcome = "asthma"  # outcome variable (numeric)
num_vars = [
    "ozone",
    "pm2_5",
    "diesel_pm",
    "drinking_water",
    "pesticides",
    "tox_release",
    "traffic",
    "cleanup_sites",
    "groundwater_threats",
    "imp_water_bodies",
    "education",
    "haz_waste",
    "solid_waste",
    "linguistic_isolation",
    "poverty",
    "unemployment",
    "housing_burden",
]  # numeric covariates

# create two categorical covariates - see report for reasoning behind choices
# Match R ifelse(): NA inputs stay NA (np.where would coerce NA comparisons to False)
def _any_flag(series: pd.Series) -> pd.Series:
    out = pd.Series(np.where(series > 0, "yes", "no"), index=series.index, dtype=object)
    return out.where(series.notna())


# Rmd always builds these from continuous waste; if the RDS is already engineered
# (haz_waste_any present, haz_waste dropped), reuse existing columns.
if "haz_waste" in dat.columns:
    dat["haz_waste_any"] = pd.Categorical(_any_flag(dat["haz_waste"]))
elif "haz_waste_any" not in dat.columns:
    raise KeyError("Need haz_waste or haz_waste_any in development data.")

if "solid_waste" in dat.columns:
    dat["solid_waste_any"] = pd.Categorical(_any_flag(dat["solid_waste"]))
elif "solid_waste_any" not in dat.columns:
    raise KeyError("Need solid_waste or solid_waste_any in development data.")

# Normalize binary indicators to yes/no strings (Rmd factor levels)
for col in ("haz_waste_any", "solid_waste_any"):
    s = dat[col]
    if pd.api.types.is_numeric_dtype(s):
        dat[col] = pd.Categorical(np.where(s.astype(float) > 0, "yes", "no"))
    else:
        mapped = (
            s.astype(str)
            .str.strip()
            .str.lower()
            .replace({"1": "yes", "0": "no", "true": "yes", "false": "no"})
        )
        dat[col] = pd.Categorical(mapped)

# group counties into regions (based largely off of air committees)
bay = [
    "Alameda",
    "Contra Costa",
    "Marin",
    "Napa",
    "San Francisco",
    "San Mateo",
    "Santa Clara",
    "Solano",
    "Sonoma",
]
sjv = [
    "San Joaquin",
    "Stanislaus",
    "Merced",
    "Madera",
    "Fresno",
    "Kings",
    "Tulare",
    "Kern",
]
socal = [
    "Imperial",
    "Los Angeles",
    "Orange",
    "Riverside",
    "San Bernardino",
    "San Diego",
    "San Luis Obispo",
    "Santa Barbara",
    "Ventura",
]

if "california_county" in dat.columns:
    county = dat["california_county"].astype(str)
    dat["region"] = pd.Categorical(
        np.select(
            [county.isin(bay), county.isin(sjv), county.isin(socal)],
            ["Bay Area", "San Joaquin Valley", "SoCal"],
            default="Other",
        )
    )
elif "region" not in dat.columns:
    raise KeyError("Need california_county or region in development data.")

# Check regions
print(dat["region"].value_counts())

# Check spelling matches your data
if "california_county" in dat.columns:
    print(
        dat.loc[dat["region"] == "Other", "california_county"]
        .value_counts()
        .sort_values(ascending=False)
    )

# Set categorical columns
cat_vars = ["region", "haz_waste_any", "solid_waste_any"]  # categorical covariates

# Set group column to explore
group_var = "region"  # grouping variable for part (d)

# convert categorical variables to factors
for v in cat_vars:
    dat[v] = pd.Categorical(dat[v])

# Restrict numeric list to columns present (engineered RDS may already have dropped some)
num_vars = [v for v in num_vars if v in dat.columns]

# Check out the data
print(dat.shape)
print(dat.info())
print(dat.describe(include="all"))
if "california_county" in dat.columns:
    print(dat["california_county"].nunique())
    print(dat["california_county"].unique())

# ===========================================================================
# (a) Data collection and data quality
# ===========================================================================
# check for duplicate rows
print(int(dat.duplicated().sum()))

# distribution of the outcome
fig, ax = plt.subplots()
ax.hist(dat[outcome].dropna(), bins=30, color="steelblue", edgecolor="white")
ax.set_title(f"Distribution of {outcome}")
_show()

# distributions of numeric covariates
long_num = dat[num_vars].melt(var_name="name", value_name="value")
g = long_num.groupby("name", sort=False)
n_facets = len(num_vars)
ncols = 4
nrows = int(np.ceil(n_facets / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
axes = np.atleast_1d(axes).ravel()
for i, (name, grp) in enumerate(g):
    axes[i].hist(grp["value"].dropna(), bins=30, color="steelblue", edgecolor="white")
    axes[i].set_title(name)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.tight_layout()
_show()

fig, ax = plt.subplots()
ax.hist(dat["pesticides"].dropna(), bins=30, color="steelblue", edgecolor="white")
ax.set_title("Distribution of pesticides")
_show()

# outliers (IQR rule, same as R sapply)
outlier_counts = {}
for v in num_vars:
    x = dat[v]
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    outlier_counts[v] = int(
        ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum(skipna=True)
    )
print(pd.Series(outlier_counts))

# how balanced are the categories?
for v in cat_vars:
    counts = dat[v].value_counts(dropna=False).rename("n").reset_index()
    counts.columns = [v, "n"]
    counts["pct"] = (100 * counts["n"] / counts["n"].sum()).round(1)
    print(counts)

# ===========================================================================
# (b) Correlation with the outcome
# ===========================================================================
# Numeric covariates — Pearson correlations
cor_outcome = {
    v: dat[[v, outcome]].dropna().corr().iloc[0, 1] for v in num_vars
}
cor_outcome = pd.Series(cor_outcome).sort_values(ascending=False)
print(cor_outcome.round(3))

fig, ax = plt.subplots(figsize=(8, 6))
order = cor_outcome.reindex(cor_outcome.abs().sort_values().index)
ax.barh(order.index.astype(str), order.values, color="steelblue")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(f"Correlation with {outcome}")
ax.set_ylabel(None)
fig.tight_layout()
_show()

# scatter + lm smooth per numeric covariate
n_facets = len(num_vars)
ncols = 4
nrows = int(np.ceil(n_facets / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
axes = np.atleast_1d(axes).ravel()
for i, v in enumerate(num_vars):
    ax = axes[i]
    sub = dat[[v, outcome]].dropna()
    ax.scatter(sub[v], sub[outcome], alpha=0.3, s=10)
    if len(sub) >= 2:
        slope, intercept, *_ = stats.linregress(sub[v], sub[outcome])
        x_line = np.linspace(sub[v].min(), sub[v].max(), 100)
        ax.plot(x_line, intercept + slope * x_line, color="red")
    ax.set_xlabel(v)
    ax.set_ylabel(outcome)
    ax.set_title(v)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.tight_layout()
_show()

# Categorical covariates
for v in cat_vars:
    levels = (
        list(dat[v].cat.categories)
        if hasattr(dat[v], "cat")
        else list(pd.unique(dat[v].dropna()))
    )
    fig, ax = plt.subplots()
    data_by_level = [dat.loc[dat[v] == lev, outcome].dropna() for lev in levels]
    bp = ax.boxplot(data_by_level, patch_artist=True)
    ax.set_xticks(range(1, len(levels) + 1))
    ax.set_xticklabels([str(lev) for lev in levels])
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.set_title(f"{outcome} by {v}")
    ax.set_xlabel(v)
    _show()

    summary = (
        dat.groupby(v, observed=True)[outcome]
        .agg(n="count", mean="mean", median="median")
        .reset_index()
    )
    print(summary)

    n_levels = len(levels)
    if n_levels == 2:
        g0 = dat.loc[dat[v] == levels[0], outcome].dropna()
        g1 = dat.loc[dat[v] == levels[1], outcome].dropna()
        print(stats.ttest_ind(g0, g1, equal_var=True, nan_policy="omit"))
    else:
        # pairwise t-tests with Bonferroni adjustment (like pairwise.t.test)
        pairs = list(combinations(levels, 2))
        raw_p = []
        labels = []
        for a, b in pairs:
            ga = dat.loc[dat[v] == a, outcome].dropna()
            gb = dat.loc[dat[v] == b, outcome].dropna()
            res = stats.ttest_ind(ga, gb, equal_var=True, nan_policy="omit")
            raw_p.append(res.pvalue)
            labels.append(f"{a} vs {b}")
        adj_p = np.minimum(1.0, np.asarray(raw_p) * len(raw_p))  # Bonferroni
        print(pd.DataFrame({"pair": labels, "p_raw": raw_p, "p_bonferroni": adj_p}))

# ===========================================================================
# (c) Correlations among covariates
# ===========================================================================
cor_mat = dat[num_vars].corr(method="pearson")  # pairwise complete by default
print(cor_mat.round(2))

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cor_mat.values, cmap="bwr", vmin=-1, vmax=1)
ax.set_xticks(range(len(num_vars)))
ax.set_yticks(range(len(num_vars)))
ax.set_xticklabels(num_vars, rotation=45, ha="right")
ax.set_yticklabels(num_vars)
for i in range(len(num_vars)):
    for j in range(len(num_vars)):
        ax.text(j, i, f"{cor_mat.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="r")
fig.tight_layout()
_show()

# drop housing burden and education  (comment retained from Rmd; done at export)

# strongest pairs
cor_long = (
    cor_mat.stack()
    .rename_axis(["Var1", "Var2"])
    .reset_index(name="Freq")
)
cor_pairs = cor_long.loc[
    cor_long["Var1"].astype(str) < cor_long["Var2"].astype(str)
].copy()
cor_pairs = cor_pairs.reindex(cor_pairs["Freq"].abs().sort_values(ascending=False).index)
print(cor_pairs.head(10))

# Relationships between categorical covariates (contingency tables + chi-square)
if len(cat_vars) > 1:
    for a, b in combinations(cat_vars, 2):
        tab = pd.crosstab(dat[a], dat[b], dropna=False)
        tab.index.name, tab.columns.name = a, b
        print(tab)
        print(tab.div(tab.sum(axis=1), axis=0).round(2))
        chi2, p, dof, expected = stats.chi2_contingency(tab)
        print(f"Chi-squared test: chi2={chi2:.4f}, df={dof}, p={p}")

# Categorical vs numeric covariates (group means)
for v in cat_vars:
    print(dat.groupby(v, observed=True)[num_vars].mean())


def test_by_group(grp: str) -> pd.DataFrame:
    """One-way ANOVA of each numeric covariate by grouping factor (like R aov)."""
    rows = []
    for v in num_vars:
        sub = dat[[grp, v]].dropna()
        groups = [g[v].to_numpy() for _, g in sub.groupby(grp, observed=True)]
        if len(groups) < 2:
            continue
        # ANOVA via scipy + eta^2 from SS decomposition
        f_stat, p_value = stats.f_oneway(*groups)
        grand_mean = float(sub[v].mean())
        n_g = sub.groupby(grp, observed=True)[v].count()
        mean_g = sub.groupby(grp, observed=True)[v].mean()
        ss_between = float(((mean_g - grand_mean) ** 2 * n_g).sum())
        fitted = sub[grp].astype(object).map(mean_g.to_dict()).astype(float)
        ss_within = float(((sub[v] - fitted) ** 2).sum())
        ss_total = ss_between + ss_within
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
        rows.append(
            {"variable": v, "F": f_stat, "p_value": p_value, "eta_sq": eta_sq}
        )
    out = pd.DataFrame(rows)
    out["p_adj"] = np.minimum(1.0, out["p_value"] * len(out))  # Bonferroni
    return out.sort_values("eta_sq", ascending=False).reset_index(drop=True)


print(test_by_group("region"))
print(test_by_group("haz_waste_any"))
print(test_by_group("solid_waste_any"))

# ===========================================================================
# (d) Subgroups
# ===========================================================================
print(dat[group_var].value_counts())

# means of the outcome and covariates in each group
print(
    dat.groupby(group_var, observed=True)[[outcome] + num_vars].mean()
)

# compare covariate distributions across groups
long_grp = dat[[group_var] + num_vars].melt(
    id_vars=[group_var], var_name="name", value_name="value"
)
n_facets = len(num_vars)
ncols = 4
nrows = int(np.ceil(n_facets / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
axes = np.atleast_1d(axes).ravel()
regions = list(dat[group_var].cat.categories) if hasattr(dat[group_var], "cat") else sorted(dat[group_var].dropna().unique())
for i, name in enumerate(num_vars):
    ax = axes[i]
    data_by_region = [
        long_grp.loc[(long_grp["name"] == name) & (long_grp[group_var] == r), "value"].dropna()
        for r in regions
    ]
    ax.boxplot(data_by_region, patch_artist=True)
    ax.set_xticks(range(1, len(regions) + 1))
    ax.set_xticklabels([str(r) for r in regions])
    ax.set_title(name)
    ax.tick_params(axis="x", rotation=30)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.tight_layout()
_show()

# does the relationship with the outcome look different by group?
n_facets = len(num_vars)
ncols = 4
nrows = int(np.ceil(n_facets / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
axes = np.atleast_1d(axes).ravel()
cmap = plt.cm.tab10
for i, v in enumerate(num_vars):
    ax = axes[i]
    for j, r in enumerate(regions):
        sub = dat.loc[dat[group_var] == r, [v, outcome]].dropna()
        color = cmap(j % 10)
        ax.scatter(sub[v], sub[outcome], alpha=0.3, s=10, color=color, label=str(r))
        if len(sub) >= 2:
            slope, intercept, *_ = stats.linregress(sub[v], sub[outcome])
            x_line = np.linspace(sub[v].min(), sub[v].max(), 50)
            ax.plot(x_line, intercept + slope * x_line, color=color, linewidth=1.5)
    ax.set_title(v)
    ax.set_xlabel(v)
    ax.set_ylabel(outcome)
axes[0].legend(fontsize=7, loc="best")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.tight_layout()
_show()

# ===========================================================================
# (e) Missing values
# ===========================================================================
vars_ = [outcome] + num_vars + cat_vars

missing = pd.DataFrame(
    {
        "variable": vars_,
        "n_missing": [int(dat[v].isna().sum()) for v in vars_],
    }
)
missing["pct_missing"] = (100 * missing["n_missing"] / len(dat)).round(1)
missing = missing.sort_values("pct_missing", ascending=False).reset_index(drop=True)
print(missing)

# rows we would keep if we dropped every incomplete row
print(int(dat[vars_].notna().all(axis=1).sum()))

# Also check for placeholder values that really mean "missing"
placeholder_hits = {}
for v in vars_:
    col = dat[v]
    # Compare element-wise against placeholders (numeric and string)
    hits = col.isin([-999, -99, "", "?", "NA"]).sum()
    placeholder_hits[v] = int(hits)
print(pd.Series(placeholder_hits))

# Is missingness related to the outcome?
for v in missing.loc[
    (missing["n_missing"] > 0) & (missing["variable"] != outcome), "variable"
]:
    print(f"\n{v}")
    miss = dat[v].isna()
    g_miss = dat.loc[miss, outcome].dropna()
    g_obs = dat.loc[~miss, outcome].dropna()
    print(stats.ttest_ind(g_miss, g_obs, equal_var=True, nan_policy="omit"))

vars_complete = [outcome] + num_vars
complete = dat[vars_complete].notna().all(axis=1)

print(int((~complete).sum()))  # how many tracts you're dropping
print((~complete).sum() / len(dat) * 100)  # percent dropped

print(dat["asthma"].mean())  # asthma mean, all tracts
print(dat.loc[complete, "asthma"].mean())  # asthma mean, complete tracts only
print(pd.crosstab(dat["region"], complete))  # are dropped tracts concentrated in one region?

# ===========================================================================
# Modeling dataset export
# ===========================================================================
# One training CSV: engineered categoricals, correlated/redundant columns dropped,
# existing fold_id kept. Covariates are left unimputed — fold-wise median
# imputation happens in modelfit.py. asthma is NOT imputed; validation_baseline
# drops NA-outcome rows before CV.

# Drop continuous waste (replaced by *_any) and highly correlated SES vars
drop_vars = ["haz_waste", "solid_waste", "education", "housing_burden"]

dat_model = dat.drop(columns=[c for c in drop_vars if c in dat.columns])

assert "fold_id" in dat_model.columns

# Report covariate missingness left for fold-wise imputation in modeling
num_export = [
    c
    for c in dat_model.columns
    if c not in ("asthma", "fold_id", "census_tract", "california_county")
    and c not in cat_vars
    and pd.api.types.is_numeric_dtype(dat_model[c])
]
missing_export = pd.DataFrame(
    {
        "variable": num_export,
        "n_missing": [int(dat_model[c].isna().sum()) for c in num_export],
    }
)
missing_export["pct_missing"] = (
    100 * missing_export["n_missing"] / len(dat_model)
).round(1)
print(missing_export.loc[missing_export["n_missing"] > 0].reset_index(drop=True))

print(
    "NA asthma rows left for validation to drop:",
    int(dat_model["asthma"].isna().sum()),
)
print(list(dat_model.columns))

# Output path — same preference as Rmd
if (src_dir / "../data/clean_data").resolve().exists() or (src_dir / "../data").resolve().exists():
    out_dir = (src_dir / "../data/clean_data").resolve()
else:
    out_dir = (cwd / "data/clean_data").resolve()
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "eda_training.csv"
# Convert categoricals to plain values for CSV (like write_csv on factors; keep NA)
dat_out = dat_model.copy()
for c in dat_out.select_dtypes(include=["category"]).columns:
    dat_out[c] = dat_out[c].astype(object)
dat_out.to_csv(out_path, index=False)
print(f"Wrote single training CSV: {out_path.resolve()}")
