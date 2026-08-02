"""
Download raw series from FRED (Federal Reserve Economic Data).

No API key is required: every FRED series has a public CSV endpoint of the form
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>

We download each series once and cache it under data/ so re-runs are fast and
the project is reproducible. Missing observations in FRED CSVs are coded "."
and are converted to NaN here.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
from curl_cffi import requests as crequests  # browser-grade TLS; more robust than plain requests for this endpoint

RAW_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR.mkdir(exist_ok=True)

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

# series_id -> (human-readable description, role in the project)
SERIES: dict[str, tuple[str, str]] = {
    # ---- target variable ----
    "DGS10":       ("10-Year Treasury yield, %", "target"),
    # ---- funding / liquidity stress ----
    "SOFR":        ("Secured Overnight Financing Rate, %", "feature"),
    "RRPONTSYD":   ("Fed ON RRP facility usage, $bn", "feature"),
    # ---- credit stress (flight-to-quality proxy) ----
    # (ICE BofA OAS series on FRED were restructured in Aug 2023 and lost
    #  their history; Moody's Baa spread gives us the same signal back to 1986)
    "BAA10Y":      ("Moody's Baa corporate yield minus 10Y Treasury, pp", "feature"),
    # ---- risk appetite ----
    "VIXCLS":      ("CBOE VIX close", "feature"),
    "DTWEXBGS":    ("Nominal broad U.S. dollar index", "feature"),
    # ---- yield-curve shape / monetary policy ----
    "T10Y2Y":      ("10Y minus 2Y Treasury spread, pp", "feature"),
    "T10Y3M":      ("10Y minus 3M Treasury spread, pp", "feature"),
    "WALCL":       ("Fed total assets (QE proxy), $mn, weekly", "feature"),
}


def fetch_one(sid: str, attempts: int = 4) -> pd.Series:
    """Fetch a single FRED series and return it as a float Series indexed by date.

    Retries with exponential backoff — the FRED endpoint occasionally stalls
    on the first request after an idle period.
    """
    for i in range(attempts):
        try:
            resp = crequests.get(
                FRED_URL.format(sid=sid),
                impersonate="chrome",
                timeout=60,
            )
            resp.raise_for_status()
            break
        except crequests.RequestsError as exc:
            if i == attempts - 1:
                raise
            wait = 2 ** i
            print(f"  retry {i + 1}/{attempts - 1} for {sid} after {type(exc).__name__} (wait {wait}s)")
            time.sleep(wait)
    df = pd.read_csv(io.StringIO(resp.text), na_values=".")
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(df[date_col]),
        name=sid,
    )
    return s.sort_index()


def main() -> None:
    panel = {}
    for sid, (desc, role) in SERIES.items():
        out = RAW_DIR / f"{sid}.csv"
        if out.exists():
            s = pd.read_csv(out, index_col=0, parse_dates=True).iloc[:, 0]
            s.name = sid
            print(f"[cache] {sid:<12} {len(s):>5} obs  {s.index.min().date()} -> {s.index.max().date()}")
        else:
            s = fetch_one(sid)
            s.to_csv(out)
            print(f"[fetch] {sid:<12} {len(s):>5} obs  {s.index.min().date()} -> {s.index.max().date()}"
                  f"   ({role}: {desc})")
            time.sleep(0.5)  # be polite to the server
        panel[sid] = s

    # Combine into one table on a common (business-day) calendar and save.
    df = pd.DataFrame(panel)
    df = df.sort_index()
    df.to_csv(RAW_DIR / "raw_panel.csv")
    print(f"\nCombined panel: {df.shape[0]} rows x {df.shape[1]} columns -> data/raw_panel.csv")
    print("\nMissing values per series (before any cleaning):")
    print(df.isna().sum().to_string())


if __name__ == "__main__":
    main()
