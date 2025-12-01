# run_final.py
# Final pipeline: temporal split → features → 3 supervised models → metrics + plots
#
# Usage (from your project folder):
#   python run_final.py \
#     --data data/working/merged.csv \
#     --year_col year \
#     --playcount_col playcount \
#     --numeric_cols duration,year,artist_familiarity,artist_hotttnesss \
#     --train_end_year 2008

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def temporal_split(df: pd.DataFrame, year_col: str, train_end_year: int):
    """
    Train = rows with year <= train_end_year
    Test  = rows with year > train_end_year
    """
    tr = df[df[year_col] <= train_end_year].copy()
    te = df[df[year_col] > train_end_year].copy()
    if tr.empty or te.empty:
        raise ValueError(
            f"Temporal split produced empty train/test. "
            f"Check --year_col={year_col} and --train_end_year={train_end_year}."
        )
    return tr, te


def build_features(train_df, test_df, numeric_cols, target_col):
    """
    Scale numeric features; log1p-transform the target (playcount).
    Returns:
      Xtr, Xte, ytr, yte
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_df[numeric_cols])
    Xte = scaler.transform(test_df[numeric_cols])

    ytr = np.log1p(train_df[target_col].to_numpy())
    yte = np.log1p(test_df[target_col].to_numpy())
    return Xtr, Xte, ytr, yte


def eval_reg(y_true, y_pred):
    """
    Compute RMSE, MAE, R2 (manual RMSE to avoid sklearn param differences).
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def plot_pred_vs_actual(y_true, y_pred, path, title):
    plt.figure()
    plt.scatter(y_true, y_pred, s=8, alpha=0.4)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("Actual log(playcount)")
    plt.ylabel("Predicted log(playcount)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_resid_hist(y_true, y_pred, path, title):
    resid = y_true - y_pred
    plt.figure()
    plt.hist(resid, bins=30)
    plt.xlabel("Residual (y - ŷ)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_feature_importances(importances, feature_names, path, title):
    """
    Bar chart of feature importances, sorted descending.
    Only used for models that expose `feature_importances_`.
    """
    importances = np.asarray(importances)
    idx = np.argsort(importances)[::-1]
    sorted_imp = importances[idx]
    sorted_names = [feature_names[i] for i in idx]

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(sorted_imp)), sorted_imp)
    plt.xticks(range(len(sorted_imp)), sorted_names, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main(args):
    # Create output folder
    figs_dir = Path("figs")
    figs_dir.mkdir(exist_ok=True)

    # Load data (CSV or Parquet)
    if args.data.lower().endswith(".csv"):
        df = pd.read_csv(args.data)
    else:
        df = pd.read_parquet(args.data)

    # Basic checks
    for c in [args.year_col, args.playcount_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column in data: {c}")

    numeric_cols = [c.strip() for c in args.numeric_cols.split(",") if c.strip()]
    for c in numeric_cols:
        if c not in df.columns:
            raise ValueError(f"Missing numeric feature column: {c}")

    # Minimal NA handling for numeric features
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))

    # Temporal split
    train_df, test_df = temporal_split(df, args.year_col, args.train_end_year)

    # Features & target (log1p(playcount))
    Xtr, Xte, ytr, yte = build_features(train_df, test_df, numeric_cols, args.playcount_col)

    # Define supervised models
    models = {
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000, random_state=42),
        "HistGB": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=600,
            random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        ),
    }

    metrics = {}
    preds = {}

    # Train + evaluate each model
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)
        rmse, mae, r2 = eval_reg(yte, yhat)
        metrics[name] = (rmse, mae, r2)
        preds[name] = yhat

        print(f"{name}: RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

        # Plots: predicted vs actual and residuals
        plot_pred_vs_actual(
            yte, yhat,
            figs_dir / f"{name}_pred_vs_actual.png",
            title=f"{name}: Predicted vs Actual (log playcount)",
        )
        plot_resid_hist(
            yte, yhat,
            figs_dir / f"{name}_residuals.png",
            title=f"{name}: Residuals (y - ŷ)",
        )

        # Feature importances for tree models
        if hasattr(model, "feature_importances_"):
            plot_feature_importances(
                model.feature_importances_,
                numeric_cols,
                figs_dir / f"{name}_feature_importances.png",
                title=f"{name}: Feature Importances",
            )

    # Save metrics summary to a TSV
    metrics_path = figs_dir / "metrics_summary.tsv"
    with metrics_path.open("w") as f:
        f.write("model\tRMSE\tMAE\tR2\n")
        for name, (rmse, mae, r2) in metrics.items():
            f.write(f"{name}\t{rmse:.4f}\t{mae:.4f}\t{r2:.4f}\n")

    print(f"\n✅ Saved plots and metrics to: {figs_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Song popularity prediction – final pipeline (supervised)")
    p.add_argument("--data", required=True,
                   help="Path to CSV or Parquet with features + playcount + year")
    p.add_argument("--year_col", default="year")
    p.add_argument("--playcount_col", default="playcount")
    p.add_argument(
        "--numeric_cols",
        required=True,
        help='Comma-separated numeric features, e.g. "duration,year,artist_familiarity,artist_hotttnesss"',
    )
    p.add_argument("--train_end_year", type=int, default=2008)
    args = p.parse_args()
    main(args)
