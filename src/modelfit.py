# Load Data

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
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

# One-hot categoricals once (closed category set); scaling stays fold-wise for linear/lasso
X_all = pd.get_dummies(df[feature_cols], columns=CAT_COLS, drop_first=False)
y_all = df[TARGET].to_numpy(dtype=float)
fold_id = df["fold_id"].to_numpy(dtype=int)
feature_names = list(X_all.columns)
X_mat = X_all.to_numpy(dtype=float)

print(f"Feature matrix: {X_mat.shape} (numeric + one-hot)")
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

for name, builder in FEATURE_SETS.items():
    Xt, _ = builder(X_all, X_all.iloc[:5])
    print(f"{name}: {Xt.shape[1]} columns")


def run_fold_cv(
    fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    name: str,
    feature_set: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Train on each fold's complement and score the held-out fold. Feature builders run inside each fold so fold-local statistics use training data only."""
    builder = FEATURE_SETS[feature_set]
    oof = np.full(len(y_all), np.nan)
    per_fold = []

    for k in range(1, 6):
        train = fold_id != k
        val = fold_id == k

        X_tr, X_va = builder(X_all.loc[train], X_all.loc[val])
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

# Lasso Regression 

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
    model_notes[key] = "OLS + StandardScaler fit on each training fold"

def fit_predict_lasso(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray
) -> np.ndarray:
    """Fit LassoCV with fold-wise StandardScaler and return validation predictions."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)
    model = LassoCV(
        cv=5,
        random_state=RANDOM_STATE,
        max_iter=10000,
        n_jobs=-1,
    )
    model.fit(X_tr, y_train)
    return model.predict(X_va)


for fs in FEATURE_SETS:
    key = f"lasso|{fs}"
    oof_preds[key], fold_rmses[key] = run_fold_cv(
        fit_predict_lasso, f"Lasso [{fs}]", fs
    )
    model_notes[key] = "LassoCV(cv=5) + StandardScaler fit on each training fold"


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
    model_notes[key] = f"RandomForestRegressor{RF_PARAMS}"