from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# The script lives in part2/; the existing data/ folder is at the repo root.
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "part2" / "results"

POLY = ["poverty", "unemployment", "linguistic_isolation", "ozone", "pm2_5"]
CATEGORICAL = ["region", "haz_waste_any", "solid_waste_any"]
PART1_CV_RMSE = 21.169666866238725

REGIONS = {
    "Bay Area": (
        "Alameda", "Contra Costa", "Marin", "Napa", "San Francisco",
        "San Mateo", "Santa Clara", "Solano", "Sonoma",
    ),
    "San Joaquin Valley": (
        "San Joaquin", "Stanislaus", "Merced", "Madera",
        "Fresno", "Kings", "Tulare", "Kern",
    ),
    "SoCal": (
        "Imperial", "Los Angeles", "Orange", "Riverside",
        "San Bernardino", "San Diego", "San Luis Obispo",
        "Santa Barbara", "Ventura",
    ),
}
COUNTY_TO_REGION = {
    county: region for region, counties in REGIONS.items() for county in counties
}


def model_matrix(data, settings):
    """Apply the feature settings learned from the development data."""
    x = pd.get_dummies(
        data[settings["features"]], columns=CATEGORICAL, dtype=float
    ).reindex(columns=settings["encoded_columns"], fill_value=0.0)
    x = x.fillna(settings["medians"])
    squares = (x[POLY] - settings["centers"]) ** 2
    return np.column_stack([x.to_numpy(dtype=float), squares.to_numpy()])


def main():
    development = pd.read_csv(
        ROOT / "data" / "clean_data" / "eda_training.csv"
    ).set_index("census_tract")
    source = pd.read_csv(
        ROOT / "data" / "calenviroscreen.csv"
    ).set_index("census_tract")

    if (
        not development.index.is_unique
        or not source.index.is_unique
        or len(development) != 6428
        or len(source) != 8035
        or not development.index.isin(source.index).all()
    ):
        raise ValueError("Inputs do not match the expected Part 1 split.")

    features = [
        c for c in development.columns
        if c not in {"asthma", "fold_id", "california_county"}
    ]
    numeric = [c for c in features if c not in CATEGORICAL]

    # Check that the development export still has the original numeric
    # values and missingness. This catches an older, pre-imputed export.
    if not np.allclose(
        development[numeric],
        source.loc[development.index, numeric],
        equal_nan=True,
    ):
        raise ValueError("Development predictors differ from the raw data.")

    train = development.dropna(subset=["asthma"])
    medians = train[numeric].median()
    settings = {
        "features": features,
        "encoded_columns": list(
            pd.get_dummies(train[features], columns=CATEGORICAL).columns
        ),
        "medians": medians,
        "centers": train[POLY].fillna(medians[POLY]).mean(),
    }

    # Refit the selected Part 1 configuration on development data only.
    forest = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(model_matrix(train, settings), train["asthma"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pipeline_path = OUTPUT / "reconstructed_b_poly_forest.joblib"
    joblib.dump({"model": forest, "settings": settings}, pipeline_path)

    # Score the original holdout with the saved development-fitted settings.
    holdout = source.loc[~source.index.isin(development.index)].copy()
    if len(holdout) != 1607:
        raise ValueError("Expected 1,607 holdout tracts.")

    holdout["region"] = (
        holdout["california_county"].map(COUNTY_TO_REGION).fillna("Other")
    )
    for col in ("haz_waste", "solid_waste"):
        holdout[f"{col}_any"] = np.where(holdout[col] > 0, "yes", "no")
        holdout.loc[holdout[col].isna(), f"{col}_any"] = np.nan

    scored = holdout.dropna(subset=["asthma"])
    saved = joblib.load(pipeline_path)
    predicted = saved["model"].predict(model_matrix(scored, saved["settings"]))
    holdout_rmse = float(np.sqrt(np.mean((scored["asthma"] - predicted) ** 2)))

    summary = pd.DataFrame([{
        "model": "Random forest (B_poly; reconstructed)",
        "development_n": len(train),
        "holdout_n_scored": len(scored),
        "part1_pooled_cv_rmse": PART1_CV_RMSE,
        "holdout_rmse": holdout_rmse,
        "holdout_minus_cv_rmse": holdout_rmse - PART1_CV_RMSE,
    }])
    summary.to_csv(OUTPUT / "holdout_prediction_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
