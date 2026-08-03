"""
Experiment 2 — Forecasting 10Y Treasury yield VOLATILITY.

Why volatility? Experiment 1 showed the *direction* of yield changes is not
predictable out-of-sample from public liquidity signals (relationships flip
sign across regimes). Volatility is different on two counts:
  1. stress indicators drive risk directly — "stress up -> vol up" does not
     depend on the policy regime, so the sign is stable;
  2. volatility clusters (today's vol is the best single predictor of
     tomorrow's), giving the model a strong backbone to build on.

Target   : log of FORWARD 21-trading-day realized volatility of daily 10Y
           yield changes (bp/day). Logs because vol is positive and skewed —
           a linear model fits log-vol far better than raw vol.
Features : 8 funding/liquidity stress z-scores (from build_features.py)
           + log backward 5d and 21d realized vol (vol clustering).
Model    : Ridge(alpha=100) on a ROLLING 3-year window (adapts to vol-regime
           level shifts), refit every 21 days.
Leakage control: a 21-row embargo — when predicting day t, training stops at
           t-21 so no training target window overlaps the prediction day.
Benchmark: naive persistence (log backward 21d vol).
Model selection used ONLY the validation segment (2020-2023); 2024-2026 is
reported as the untouched test segment.

Outputs:
  data/vol_predictions.csv
  output/vol_pred_vs_actual.png
  output/vol_coef_paths.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "output"
OUT.mkdir(exist_ok=True)

HORIZON = 21        # forward vol window (trading days)
REFIT = 21          # monthly refit
INIT = 504          # ~2 years before first OOS prediction
WINDOW = 756        # rolling 3-year training window
ALPHA = 100.0       # shrinkage scaled to n (heuristic alpha ~ 0.1 * n)
VAL_END = pd.Timestamp("2023-12-31")


def build_xy() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (features, log forward 21d vol, forward 21d vol in bp/day)."""
    ds = pd.read_csv(DATA / "dataset.csv", index_col=0, parse_dates=True)
    z = ds.drop(columns=["fwd_chg_bp"])
    dgs10 = pd.read_csv(DATA / "raw_panel.csv", index_col=0, parse_dates=True)["DGS10"].dropna()
    d1 = dgs10.diff() * 100.0  # daily change, bp

    fwd_vol = d1.shift(-1).rolling(HORIZON).std().shift(-(HORIZON - 1))  # t+1..t+H
    back_vol5 = d1.rolling(5).std()
    back_vol21 = d1.rolling(21).std()

    X = z.copy()
    X["log_bvol5"] = np.log(back_vol5.reindex(z.index))
    X["log_bvol21"] = np.log(back_vol21.reindex(z.index))
    y_log = np.log(fwd_vol.reindex(z.index))

    df = X.join(y_log.rename("y_log")).join(fwd_vol.reindex(z.index).rename("fwd_vol")).dropna()
    return df.drop(columns=["y_log", "fwd_vol"]), df["y_log"], df["fwd_vol"]


def walk_forward(X: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Rolling-window walk-forward with embargo; returns (pred, coef path)."""
    pred = pd.Series(np.nan, index=y.index, dtype=float)
    coef_path = []
    for s in range(INIT, len(y), REFIT):
        e = min(s + REFIT, len(y))
        lo = max(0, s - HORIZON - WINDOW)           # rolling 3y window
        hi = s - HORIZON                            # embargo: targets must not overlap day s
        m = Ridge(alpha=ALPHA).fit(X.iloc[lo:hi], y.iloc[lo:hi])
        pred.iloc[s:e] = m.predict(X.iloc[s:e])
        coef_path.append(pd.Series(m.coef_, index=X.columns, name=y.index[s]))
    return pred, pd.DataFrame(coef_path)


def metrics(y: pd.Series, p: pd.Series, naive: pd.Series) -> dict:
    r2_mean = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r2_naive = 1 - ((y - p) ** 2).sum() / ((y - naive) ** 2).sum()
    return {"r2_vs_mean": r2_mean, "r2_vs_naive": r2_naive,
            "corr": np.corrcoef(p, y)[0, 1], "n": len(y)}


def main() -> None:
    X, y_log, fwd_vol = build_xy()

    pred, coefs = walk_forward(X, y_log)
    naive = X["log_bvol21"].reindex(pred.index)

    rows = []
    oos = pred.notna()  # exclude pre-OOS warmup rows (pred is NaN there)
    for seg, mask in [("validation 2020-2023", pred.index <= VAL_END),
                      ("test 2024-2026", pred.index > VAL_END)]:
        m = oos & mask
        r = metrics(y_log[pred.index][m], pred[m], naive[m])
        r["segment"] = seg
        rows.append(r)
    res = pd.DataFrame(rows).set_index("segment")
    pd.set_option("display.float_format", lambda v: f"{v:+.4f}")
    print(f"OOS period: {pred.first_valid_index().date()} -> {pred.last_valid_index().date()}")
    print(res.to_string())

    out = pd.DataFrame({
        "fwd_vol_bp": fwd_vol.reindex(pred.index),          # realized, bp/day
        "pred_logvol": pred,
        "pred_vol_bp": np.exp(pred),                        # model, bp/day
        "naive_vol_bp": np.exp(naive),
    })
    out.to_csv(DATA / "vol_predictions.csv")

    print("\nfinal-window coefficients (log-vol per unit):")
    print(coefs.iloc[-1].sort_values().round(3).to_string())

    # ---- figure: predicted vs realized vol (test segment zoom) ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    ax = axes[0]
    ax.plot(out.index, out["fwd_vol_bp"], lw=0.7, alpha=0.6, label="realized 21d vol")
    ax.plot(out.index, out["pred_vol_bp"], lw=1.2, label="ridge forecast")
    ax.plot(out.index, out["naive_vol_bp"], lw=0.9, alpha=0.7, label="naive persistence")
    ax.axvline(VAL_END, color="k", ls="--", lw=0.8)
    ax.text(VAL_END, ax.get_ylim()[1] * 0.95, " test ->", fontsize=8, va="top")
    ax.set_ylabel("bp/day"); ax.legend(); ax.set_title("Forward 21d realized vol: forecast vs realized")
    ax = axes[1]
    corr126 = out["pred_vol_bp"].rolling(126).corr(out["fwd_vol_bp"])
    ax.plot(corr126.index, corr126, lw=1.0)
    ax.axhline(0, c="k", lw=0.6)
    ax.axhline(corr126.mean(), c="r", ls="--", lw=0.8, label=f"mean {corr126.mean():+.3f}")
    ax.legend(); ax.set_title("Rolling 126-day forecast-realized correlation")
    fig.tight_layout(); fig.savefig(OUT / "vol_pred_vs_actual.png", dpi=150); plt.close(fig)

    # ---- figure: coefficient paths ----
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in coefs.columns:
        ax.plot(coefs.index, coefs[c], lw=0.9, label=c)
    ax.axhline(0, c="k", lw=0.6)
    ax.legend(ncol=5, fontsize=7)
    ax.set_title(f"Ridge coefficients over time (rolling 3y, alpha={ALPHA:g})")
    fig.tight_layout(); fig.savefig(OUT / "vol_coef_paths.png", dpi=150); plt.close(fig)

    print("\nfigures -> output/vol_pred_vs_actual.png, output/vol_coef_paths.png")


if __name__ == "__main__":
    main()
