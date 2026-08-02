"""
Application: volatility-targeted duration sizing.

The vol forecast (Experiment 2) is used the way a rates desk would actually
use it: size a long 10Y-Treasury position inversely to predicted volatility,
so risk exposure stays roughly constant through time.

    weight_t = clip(TARGET_VOL / predicted_vol_t, 0, W_MAX), executed with a
    1-day lag (today's position uses yesterday's forecast).

Strategies compared (all on the out-of-sample period only):
    1. buy & hold        : w = 1
    2. vol-target (model): sized by the ridge forecast
    3. vol-target (naive): sized by naive persistence (honest baseline)

Daily total return of a 10Y note approximated from the yield series:
    ret_t = yield_t / 252  (carry)  -  DURATION * dy_t  (price change)
with DURATION = 8 (modified duration of a 10Y note). Transaction cost of
COST_BP per unit of absolute weight change is charged to sized strategies.

Outputs: output/backtest.png, data/backtest_returns.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "output"

DURATION = 8.0        # modified duration of a 10Y note
TARGET_VOL = 5.0      # bp/day — typical mid-regime level
W_MAX = 2.0           # leverage cap
COST_BP = 1.0 / 1e4   # 1 bp per unit |dw|, in return units
ANN = 252


def daily_10y_return(y_pct: pd.Series) -> pd.Series:
    """Approximate daily total return of a 10Y note from the yield (in %)."""
    dy = y_pct.diff() / 100.0                    # decimal change
    carry = y_pct / 100.0 / ANN                  # daily yield accrual
    return (carry - DURATION * dy).rename("ret")


def perf_stats(ret: pd.Series) -> dict:
    ann_ret = ret.mean() * ANN
    ann_vol = ret.std() * np.sqrt(ANN)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    eq = (1 + ret).cumprod()
    max_dd = (eq / eq.cummax() - 1).min()
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}


def main() -> None:
    pred = pd.read_csv(DATA / "vol_predictions.csv", index_col=0, parse_dates=True)
    dgs10 = pd.read_csv(DATA / "raw_panel.csv", index_col=0, parse_dates=True)["DGS10"].dropna()

    ret = daily_10y_return(dgs10).reindex(pred.index)
    df = pred.join(ret).dropna()

    w_model = (TARGET_VOL / df["pred_vol_bp"]).clip(0, W_MAX).shift(1)
    w_naive = (TARGET_VOL / df["naive_vol_bp"]).clip(0, W_MAX).shift(1)

    strat = pd.DataFrame({
        "buy_hold": df["ret"],
        "vol_target_model": w_model * df["ret"] - COST_BP * w_model.diff().abs(),
        "vol_target_naive": w_naive * df["ret"] - COST_BP * w_naive.diff().abs(),
    }).dropna()
    strat.to_csv(DATA / "backtest_returns.csv")

    stats = pd.DataFrame({c: perf_stats(strat[c]) for c in strat.columns}).T
    # risk-stability: how tightly does each strategy's rolling 63d realized vol
    # track its own mean? (std of rolling vol — lower = more predictable risk)
    roll_vol = strat.rolling(63).std() * np.sqrt(ANN)
    stats["vol_of_vol"] = roll_vol.std()
    pd.set_option("display.float_format", lambda v: f"{v:+.3f}")
    print(f"backtest: {strat.index.min().date()} -> {strat.index.max().date()} "
          f"(OOS only, costs {COST_BP * 1e4:.0f} bp per unit turnover)")
    print(stats.to_string())
    print(f"\navg weight model {w_model.mean():.2f}, naive {w_naive.mean():.2f}; "
          f"daily turnover model {w_model.diff().abs().mean():.3f}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    ax = axes[0]
    eq = (1 + strat).cumprod()
    for c, lab in [("buy_hold", "buy & hold 10Y"),
                   ("vol_target_naive", "vol-target (naive)"),
                   ("vol_target_model", "vol-target (ridge forecast)")]:
        ax.plot(eq.index, eq[c], lw=1.1, label=lab)
    ax.axhline(1, c="k", lw=0.5)
    ax.set_ylabel("growth of $1"); ax.legend()
    ax.set_title("Vol-targeted 10Y duration vs buy & hold (out-of-sample, net of costs)")
    ax = axes[1]
    ax.plot(w_model.index, w_model, lw=0.8, label="weight (model)", color="C2")
    ax.plot(w_naive.index, w_naive, lw=0.8, alpha=0.6, label="weight (naive)", color="C1")
    ax.axhline(1, c="k", lw=0.5)
    ax.set_ylabel("position weight"); ax.legend()
    ax = axes[2]
    ax.plot(roll_vol.index, roll_vol["buy_hold"], lw=0.9, label="realized 63d vol: buy & hold")
    ax.plot(roll_vol.index, roll_vol["vol_target_model"], lw=0.9, label="realized 63d vol: vol-target (model)")
    ax.set_ylabel("ann. vol"); ax.legend()
    ax.set_title("Risk stability: sized exposure keeps realized vol in a tighter band")
    fig.tight_layout()
    fig.savefig(OUT / "backtest.png", dpi=150)
    plt.close(fig)
    print("figure -> output/backtest.png")


if __name__ == "__main__":
    main()
