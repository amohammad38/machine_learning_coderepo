# run_midterm.py
# Minimal midterm pipeline: temporal split → features → ElasticNet + Tree → metrics + plots
# Usage (from your project folder):
#   python run_midterm.py \
#     --data data/working/merged.csv \
#     --year_col year \
#     --playcount_col playcount \
#     --numeric_cols tempo,loudness,duration,year \
#     --train_end_year 2008
#
# Input file must have at least: year, playcount, and the numeric_cols you pass.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def temporal_split(df: pd.DataFrame, year_col: str, train_end_year: int):
    """Train = rows with year <= train_end_year; Test = rows with year > train_end_year."""
    tr = df[df[year_col] <= train_end_year].copy()
    te = df[df[year_col] > train_end_year].copy()
    if tr.empty or te.empty:
        raise ValueError(
            f"Temporal split produced empty train/test. "
            f"Check --year_col={year_col} and --train_end_year={train_end_year}."
        )
    return tr, te


def build_features(train_df, test_df, numeric_cols, target_col):
    """Scale numeric features; log1p-transform the target (playcount)."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_df[numeric_cols])
    Xte = scaler.transform(test_df[numeric_cols])
    ytr = np.log1p(train_df[target_col].to_numpy())
    yte = np.log1p(test_df[target_col].to_numpy())
    return Xtr, Xte, ytr, yte


def eval_reg(y_true, y_pred, label):
    """Print and return RMSE, MAE, R2 (manual RMSE to avoid sklearn param differences)."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label}: RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    return rmse, mae, r2


def plot_pred_vs_actual(y_true, y_pred, path):
    plt.figure()
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("Actual log(playcount)")
    plt.ylabel("Predicted log(playcount)")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_resid_hist(y_true, y_pred, path):
    resid = y_true - y_pred
    plt.figure()
    plt.hist(resid, bins=30)
    plt.xlabel("Residual (y - ŷ)")
    plt.ylabel("Count")
    plt.title("Residuals")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main(args):
    # Create output folder
    Path("figs").mkdir(exist_ok=True)

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

    # Models
    enet = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000, random_state=42)
    tree = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=600, random_state=42)

    # Train + evaluate
    enet.fit(Xtr, ytr)
    pred_en = enet.predict(Xte)
    eval_reg(yte, pred_en, "ElasticNet")

    tree.fit(Xtr, ytr)
    pred_tr = tree.predict(Xte)
    eval_reg(yte, pred_tr, "Tree(HistGB)")

    # Plots
    plot_pred_vs_actual(yte, pred_tr, "figs/reg_pred_actual_scatter.png")
    plot_resid_hist(yte, pred_tr, "figs/reg_resid_hist.png")

    # Save metrics summary
    with open("figs/metrics_summary.txt", "w") as f:
        def write_line(name, yhat):
            mse = mean_squared_error(yte, yhat)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(yte, yhat)
            r2 = r2_score(yte, yhat)
            f.write(f"{name}\tRMSE\t{rmse:.4f}\tMAE\t{mae:.4f}\tR2\t{r2:.4f}\n")
        write_line("ElasticNet", pred_en)
        write_line("HistGB", pred_tr)

    print("Saved plots to figs/ and metrics to figs/metrics_summary.txt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CS 4741 midterm minimal pipeline")
    p.add_argument("--data", required=True, help="Path to CSV or Parquet with features + playcount + year")
    p.add_argument("--year_col", default="year")
    p.add_argument("--playcount_col", default="playcount")
    p.add_argument("--numeric_cols", required=True, help='Comma-separated numeric features, e.g. "tempo,loudness,duration,year"')
    p.add_argument("--train_end_year", type=int, default=2008)
    args = p.parse_args()
    main(args)

