# Load Data

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def rmse(actual: np.ndarray | Sequence[float], predicted: np.ndarray | Sequence[float]) -> float:
    """Return root mean squared error between actual and predicted values."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def resolve_path(candidates: Sequence[Path | str]) -> Path:
    """Return the first candidate path that exists on disk."""
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of these paths exist:\n  - " + "\n  - ".join(map(str, candidates))
    )

csv_path = resolve_path(
    [
        Path("../data/clean_data/eda_training.csv"),
        Path("data/clean_data/eda_training.csv"),
    ]
)

raw = pd.read_csv(csv_path)
print(f"Loaded {csv_path.resolve()}")
print(f"Raw shape: {raw.shape}")

# Same NA-asthma drop as validation_baseline.R (before any CV)
asthma = pd.to_numeric(raw["asthma"], errors="coerce")
n_before = len(raw)
df = raw.loc[asthma.notna()].copy()
df["asthma"] = asthma.loc[asthma.notna()].astype(float)
df["fold_id"] = df["fold_id"].astype(int)

print(f"Dropped NA asthma: {n_before - len(df)} rows")
print(f"Eligible shape: {df.shape}")
print("Fold counts:\n", df["fold_id"].value_counts().sort_index())

assert not df["asthma"].isna().any()
assert set(df["fold_id"]) == {1, 2, 3, 4, 5}

ID_COLS = ["census_tract", "california_county", "fold_id"]
TARGET = "asthma"
CAT_COLS = ["region", "haz_waste_any", "solid_waste_any"]
CAT_COLS = [c for c in CAT_COLS if c in df.columns]

feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET]]
num_cols = [c for c in feature_cols if c not in CAT_COLS]

# One-hot categoricals once (closed category set). Numeric covariates stay
# unimputed here; medians are fit inside each outer training fold.
X_all = pd.get_dummies(df[feature_cols], columns=CAT_COLS, drop_first=False)
y_all = df[TARGET].to_numpy(dtype=float)
fold_id = df["fold_id"].to_numpy(dtype=int)
feature_names = list(X_all.columns)
# Original numeric predictors only (exclude one-hot columns)
num_feature_cols = [c for c in num_cols if c in X_all.columns]

print(f"Feature matrix: {X_all.shape} (numeric + one-hot; covariates unimputed)")
print(f"Numeric cols to median-impute fold-wise: {num_feature_cols}")
print("Features:", feature_names)

# Containers filled by model sections; evaluation section consumes these
oof_preds = {}
fold_rmses = {}
model_notes = {}

# Feature Sets

POLY_COLS = ["poverty", "unemployment", "linguistic_isolation", "ozone", "pm2_5"]
missing = [c for c in POLY_COLS if c not in X_all.columns]
assert not missing, f"POLY_COLS missing from data: {missing}"


def build_A_base(X_train: pd.DataFrame, X_val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return Set A matrices: raw predictors plus one-hot categoricals."""
    return X_train.to_numpy(dtype=float), X_val.to_numpy(dtype=float)


def build_B_poly(X_train: pd.DataFrame, X_val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return Set B matrices: Set A plus squared mean-centered POLY_COLS. Means are estimated from the training fold only."""
    mu = X_train[POLY_COLS].mean()

    def augment(X: pd.DataFrame) -> pd.DataFrame:
        """Append squared mean-centered POLY_COLS columns to X."""
        sq = (X[POLY_COLS] - mu) ** 2
        sq.columns = [f"{c}_sq" for c in POLY_COLS]
        return pd.concat([X, sq], axis=1)

    return augment(X_train).to_numpy(dtype=float), augment(X_val).to_numpy(dtype=float)


FEATURE_SETS = {"A_base": build_A_base, "B_poly": build_B_poly}

# Alpha grid roughly matching LassoCV defaults (n_alphas=100 over a log range)
LASSO_ALPHAS = np.logspace(-4, 1, 100)


def impute_numeric_fold(
    X_train: pd.DataFrame, X_val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median-impute all numeric covariates using training-fold medians only."""
    medians = X_train[num_feature_cols].median(numeric_only=True)
    X_tr = X_train.copy()
    X_va = X_val.copy()
    X_tr[num_feature_cols] = X_tr[num_feature_cols].fillna(medians)
    X_va[num_feature_cols] = X_va[num_feature_cols].fillna(medians)
    assert not X_tr[num_feature_cols].isna().any().any()
    assert not X_va[num_feature_cols].isna().any().any()
    return X_tr, X_va


for name, builder in FEATURE_SETS.items():
    X_demo_tr, _ = impute_numeric_fold(X_all, X_all.iloc[:5])
    Xt, _ = builder(X_demo_tr, X_demo_tr.iloc[:5])
    print(f"{name}: {Xt.shape[1]} columns")


def run_fold_cv(
    fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    name: str,
    feature_set: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Train on each fold's complement and score the held-out fold.

    Per outer fold: train-only median imputation → feature builder (poly means
    from train only) → model fit/predict. Scaling for linear/lasso is handled
    inside each fit_predict_fn.
    """
    builder = FEATURE_SETS[feature_set]
    oof = np.full(len(y_all), np.nan)
    per_fold = []

    for k in range(1, 6):
        train = fold_id != k
        val = fold_id == k

        X_tr_imp, X_va_imp = impute_numeric_fold(X_all.loc[train], X_all.loc[val])
        X_tr, X_va = builder(X_tr_imp, X_va_imp)
        y_hat_val = fit_predict_fn(X_tr, y_all[train], X_va)
        oof[val] = y_hat_val
        fold_rmse = rmse(y_all[val], y_hat_val)
        per_fold.append({"fold": k, "validation_n": int(val.sum()), "validation_rmse": fold_rmse})
        print(f"{name} | fold {k}: RMSE={fold_rmse:.4f} (n={int(val.sum())})")

    assert not np.isnan(oof).any()
    fold_table = pd.DataFrame(per_fold)
    print(
        f"{name} | mean fold RMSE={fold_table['validation_rmse'].mean():.4f} | "
        f"pooled OOF RMSE={rmse(y_all, oof):.4f}"
    )
    return oof, fold_table


# Linear / Lasso / RF

def fit_predict_linear(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray
) -> np.ndarray:
    """Fit OLS with fold-wise StandardScaler and return validation predictions."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)
    model = LinearRegression()
    model.fit(X_tr, y_train)
    return model.predict(X_va)


for fs in FEATURE_SETS:
    key = f"linear|{fs}"
    oof_preds[key], fold_rmses[key] = run_fold_cv(
        fit_predict_linear, f"Linear [{fs}]", fs
    )
    model_notes[key] = (
        "OLS + train-fold median impute + StandardScaler fit on each training fold"
    )


def fit_predict_lasso(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray
) -> np.ndarray:
    """Tune Lasso alpha with StandardScaler inside each inner CV fold."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", Lasso(max_iter=10000, random_state=RANDOM_STATE)),
        ]
    )
    search = GridSearchCV(
        pipe,
        param_grid={"lasso__alpha": LASSO_ALPHAS},
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.predict(X_val)


for fs in FEATURE_SETS:
    key = f"lasso|{fs}"
    oof_preds[key], fold_rmses[key] = run_fold_cv(
        fit_predict_lasso, f"Lasso [{fs}]", fs
    )
    model_notes[key] = (
        "Pipeline(StandardScaler, Lasso) + GridSearchCV(cv=5) so scaling "
        "is re-fit inside each inner fold; train-fold median impute outside"
    )


# Random Forest

def fit_predict_rf(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray
) -> np.ndarray:
    """Fit RandomForestRegressor with RF_PARAMS and return validation predictions."""
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model.predict(X_val)


print("RF params:", RF_PARAMS)
for fs in FEATURE_SETS:
    key = f"random_forest|{fs}"
    oof_preds[key], fold_rmses[key] = run_fold_cv(
        fit_predict_rf, f"RandomForest [{fs}]", fs
    )
    model_notes[key] = (
        f"RandomForestRegressor{RF_PARAMS}; train-fold median impute; unscaled"
    )


# Summary: RMSE by model × feature set
summary_rows = []
for key in oof_preds:
    model, feature_set = key.split("|", 1)
    summary_rows.append(
        {
            "model": model,
            "feature_set": feature_set,
            "mean_fold_rmse": float(fold_rmses[key]["validation_rmse"].mean()),
            "pooled_oof_rmse": rmse(y_all, oof_preds[key]),
            "notes": model_notes.get(key, ""),
        }
    )

summary = pd.DataFrame(summary_rows).sort_values(
    ["pooled_oof_rmse", "model", "feature_set"]
).reset_index(drop=True)

print("\n" + "=" * 72)
print("CV RMSE summary (model × feature set)")
print("=" * 72)
print(
    summary[["model", "feature_set", "mean_fold_rmse", "pooled_oof_rmse"]].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)
print("=" * 72)