"""
Build the modeling dataset from the raw FRED panel.

Pipeline (every step is leakage-free by construction):
  1. Align calendars        -> keep only days when the 10Y yield (target) trades
  2. Forward-fill features  -> bridge holidays / weekly series, max 7 days stale
  3. Rolling z-scores       -> each signal = "how many std-devs vs its own
                               trailing 30-day norm" (trailing = past data only)
  4. Publication lags       -> all features lagged 1 day; WALCL (weekly release,
                               published ~1 week late) lagged an extra 7 days
  5. Target                 -> change in DGS10 over the NEXT 5 trading days,
                               in basis points

Output: data/dataset.csv  (one row per trading day: 8 features + target)
        output/signals_overview.png
        output/feature_correlation.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "output"
OUT.mkdir(exist_ok=True)

Z_WINDOW = 30        # rolling window for daily series (trading days)
WALCL_Z_WINDOW = 13  # rolling window for the weekly balance-sheet series (weeks)
HORIZON = 5          # forecast horizon in trading days
LAG = 1              # global publication lag (days)
WALCL_EXTRA_LAG = 7  # Fed balance sheet is released ~1 week after as-of date


def rolling_z(s: pd.Series, window: int) -> pd.Series:
    """(x - trailing mean) / trailing std, requiring a FULL window of history."""
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=1)
    return (s - mu) / sd


def main() -> None:
    df = pd.read_csv(DATA / "raw_panel.csv", index_col=0, parse_dates=True).sort_index()

    # 1. trade on the target's calendar only
    df = df[df["DGS10"].notna()].copy()

    # 2. forward-fill features across holidays (never backward -> no future data)
    feats_raw = df.drop(columns=["DGS10"]).ffill(limit=7)

    # 3. rolling z-scores
    z: dict[str, pd.Series] = {}
    for col in feats_raw.columns:
        if col == "WALCL":
            # weekly series: compute the z-score on weekly observations
            # (13 weeks ~ one quarter), then map back to daily
            weekly = feats_raw[col].resample("W-WED").last().dropna()
            z_weekly = rolling_z(weekly, WALCL_Z_WINDOW)
            z[col] = z_weekly.reindex(feats_raw.index).ffill()
        else:
            z[col] = rolling_z(feats_raw[col], Z_WINDOW)
    X = pd.DataFrame(z)

    # 4. publication lags: WALCL first (its own extra lag), then everything 1 day
    X["WALCL"] = X["WALCL"].shift(WALCL_EXTRA_LAG)
    X = X.shift(LAG)

    # 5. target: forward 5-trading-day change in the 10Y yield, in basis points
    y_bp = (df["DGS10"].shift(-HORIZON) - df["DGS10"]) * 100.0

    dataset = X.copy()
    dataset["fwd_chg_bp"] = y_bp
    dataset = dataset.dropna()  # warmup rows (first ~45d) + last 5 rows have no y
    dataset.to_csv(DATA / "dataset.csv")

    # ---- console summary ----
    feats = X.columns.tolist()
    print(f"dataset: {dataset.shape[0]} rows x {dataset.shape[1]} cols "
          f"({dataset.index.min().date()} -> {dataset.index.max().date()})")
    print(f"target mean {dataset['fwd_chg_bp'].mean():+.2f} bp, "
          f"std {dataset['fwd_chg_bp'].std():.2f} bp\n")

    # first look: full-sample correlation of each signal with the target.
    # (descriptive only — the honest test is the out-of-sample walk-forward)
    corr = dataset[feats + ["fwd_chg_bp"]].corr()["fwd_chg_bp"].drop("fwd_chg_bp")
    print("full-sample corr(signal, fwd 5d change)  [in-sample peek, NOT OOS]:")
    print(corr.sort_values().round(3).to_string())

    # ---- figure 1: signal overview (3x3 grid) ----
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    for ax, col in zip(axes.ravel(), feats + ["fwd_chg_bp"]):
        ax.plot(dataset.index, dataset[col], lw=0.7)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_title(col if col != "fwd_chg_bp" else "TARGET: fwd 5d chg (bp)", fontsize=10)
    fig.suptitle("Signals (30d rolling z-scores, lagged) and target", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "signals_overview.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: feature correlation heatmap (multicollinearity check) ----
    c = dataset[feats].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)), feats, rotation=45, ha="right")
    ax.set_yticks(range(len(feats)), feats)
    for i in range(len(feats)):
        for j in range(len(feats)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, shrink=0.8)
    ax.set_title("Feature correlation (why Ridge > OLS later)")
    fig.tight_layout()
    fig.savefig(OUT / "feature_correlation.png", dpi=150)
    plt.close(fig)
    print(f"\nfigures -> output/signals_overview.png, output/feature_correlation.png")


if __name__ == "__main__":
    main()
