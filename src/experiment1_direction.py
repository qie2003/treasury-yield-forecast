"""
Walk-forward (expanding-window) out-of-sample forecasting.

Design:
  * Expanding training window, initial length 504 trading days (~2 years),
    refit every 21 trading days (~monthly), predict the next month.
  * Anti-leakage: when predicting day t, the model is trained only on rows
    whose 5-day-forward target was already REALIZED by day t — i.e. we drop
    the last HORIZON rows of every training window.
  * Two models are walked in parallel: OLS (baseline) and Ridge (alpha=1.0).
    Features are already rolling z-scores, so inputs are roughly standardized
    and alpha=1.0 is a sensible neutral shrinkage level (tuning = future work).

Evaluation on the pooled out-of-sample record:
  * OOS R^2 vs the naive zero-change benchmark
  * directional accuracy (sign hit rate)
  * Pearson and rank correlations between predictions and realized changes
  * Newey-West (HAC, 5 lags) t-stat on the predictive regression slope
    (targets overlap by 5 days, so plain t-stats overstate significance)

Outputs:
  data/direction_predictions.csv        (y, ols_pred, ridge_pred per OOS day)
  output/direction_pred_vs_actual.png
  output/direction_rolling_ic.png
  output/direction_coef_paths.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "output"
OUT.mkdir(exist_ok=True)

HORIZON = 5      # must match build_features.py
INIT = 504       # initial training window, ~2 years of trading days
REFIT = 21       # refit frequency, ~1 month of trading days
RIDGE_ALPHA = 1.0


def walk_forward(X: pd.DataFrame, y: pd.Series, make_model) -> tuple[pd.Series, pd.DataFrame]:
    """Expanding-window walk-forward. Returns (OOS predictions, coefficient path)."""
    pred = pd.Series(np.nan, index=y.index, dtype=float)
    coef_path = []
    for s in range(INIT, len(y), REFIT):
        e = min(s + REFIT, len(y))
        # drop the last HORIZON training rows: their targets are unrealized at day s
        m = make_model()
        m.fit(X.iloc[: s - HORIZON], y.iloc[: s - HORIZON])
        pred.iloc[s:e] = m.predict(X.iloc[s:e])
        coef_path.append(pd.Series(m.coef_, index=X.columns, name=y.index[s]))
    return pred, pd.DataFrame(coef_path)


def evaluate(y: pd.Series, pred: pd.Series, label: str) -> dict:
    m = pred.notna()
    yt, pt = y[m], pred[m]
    r2_os = 1.0 - ((yt - pt) ** 2).sum() / (yt**2).sum()
    hit = (np.sign(yt) == np.sign(pt)).mean()
    corr = stats.pearsonr(pt, yt).statistic
    rank_corr = stats.spearmanr(pt, yt).statistic
    # Newey-West t-stat on slope of y ~ pred (HAC with 5 lags for 5-day overlap)
    res = sm.OLS(yt.to_numpy(), sm.add_constant(pt.to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": HORIZON}
    )
    row = dict(model=label, n=len(yt), r2_os=r2_os, hit_rate=hit,
               corr=corr, rank_corr=rank_corr, nw_tstat=res.tvalues[1])
    return row


def main() -> None:
    ds = pd.read_csv(DATA / "dataset.csv", index_col=0, parse_dates=True)
    X = ds.drop(columns=["fwd_chg_bp"])
    y = ds["fwd_chg_bp"]

    pred_ols, _ = walk_forward(X, y, LinearRegression)
    pred_ridge, coefs = walk_forward(X, y, lambda: Ridge(alpha=RIDGE_ALPHA))

    results = pd.DataFrame([
        evaluate(y, pred_ols, f"OLS"),
        evaluate(y, pred_ridge, f"Ridge(a={RIDGE_ALPHA})"),
    ]).set_index("model")
    pd.set_option("display.float_format", lambda v: f"{v:+.4f}" if abs(v) < 10 else f"{v:.0f}")
    print(f"OOS period: {pred_ridge.first_valid_index().date()} -> {pred_ridge.last_valid_index().date()}")
    print(results.to_string())

    out = pd.DataFrame({"y": y, "pred_ols": pred_ols, "pred_ridge": pred_ridge}).dropna()
    out.to_csv(DATA / "direction_predictions.csv")

    # final full-sample ridge fit, for economic interpretation of coefficients
    final = Ridge(alpha=RIDGE_ALPHA).fit(X.iloc[:-HORIZON], y.iloc[:-HORIZON])
    print("\nfinal-model coefficients (bp per +1 z-score):")
    print(pd.Series(final.coef_, index=X.columns).sort_values().round(3).to_string())

    m = out.index
    # ---- figure 3: predicted vs actual ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(out["pred_ridge"], out["y"], s=4, alpha=0.3)
    lim = np.nanpercentile(np.abs(out["y"]), 99)
    ax.set_xlim(-lim / 2, lim / 2); ax.set_ylim(-lim, lim)
    ax.axhline(0, c="k", lw=0.5); ax.axvline(0, c="k", lw=0.5)
    ax.set_xlabel("predicted fwd 5d change (bp)"); ax.set_ylabel("realized (bp)")
    ax.set_title("Ridge: prediction vs realization")
    ax = axes[1]
    seg = out.loc["2024-01-01":]
    ax.plot(seg.index, seg["y"], lw=0.8, label="realized", alpha=0.8)
    ax.plot(seg.index, seg["pred_ridge"], lw=0.8, label="predicted", alpha=0.8)
    ax.axhline(0, c="k", lw=0.5); ax.legend(); ax.set_title("since 2024 (zoom)")
    fig.tight_layout(); fig.savefig(OUT / "direction_pred_vs_actual.png", dpi=150); plt.close(fig)

    # ---- figure 4: rolling 126-day forecast-realized correlation ----
    roll_ic = out["pred_ridge"].rolling(126).corr(out["y"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(roll_ic.index, roll_ic, lw=1.0)
    ax.axhline(0, c="k", lw=0.6)
    ax.axhline(roll_ic.mean(), c="r", ls="--", lw=0.8, label=f"mean {roll_ic.mean():+.3f}")
    ax.legend(); ax.set_title("Rolling 126-day forecast-realized correlation (Ridge)")
    fig.tight_layout(); fig.savefig(OUT / "direction_rolling_corr.png", dpi=150); plt.close(fig)

    # ---- figure 5: ridge coefficient path (regime shifts) ----
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in coefs.columns:
        ax.plot(coefs.index, coefs[c], lw=0.9, label=c)
    ax.axhline(0, c="k", lw=0.6)
    ax.legend(ncol=4, fontsize=8); ax.set_title(f"Ridge coefficients over time (alpha={RIDGE_ALPHA})")
    fig.tight_layout(); fig.savefig(OUT / "direction_coef_paths.png", dpi=150); plt.close(fig)

    print("\nfigures -> output/direction_pred_vs_actual.png, direction_rolling_corr.png, direction_coef_paths.png")


if __name__ == "__main__":
    main()
