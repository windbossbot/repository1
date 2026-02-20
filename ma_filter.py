from __future__ import annotations

import math
import sys
from typing import Iterable, List, Optional, Tuple

import FinanceDataReader as fdr
import pandas as pd


def parse_codes(lines: Iterable[str]) -> List[str]:
    codes: List[str] = []
    seen = set()
    for raw in lines:
        text = raw.strip()
        if not text:
            continue
        if text.isdigit() and len(text) <= 6:
            code = text.zfill(6)
            if code not in seen:
                seen.add(code)
                codes.append(code)
    return codes


def fetch_ohlcv(code: str) -> Optional[pd.DataFrame]:
    for symbol in (code, f"KRX:{code}"):
        try:
            df = fdr.DataReader(symbol)
        except Exception:
            continue
        if df is None or df.empty or "Close" not in df.columns:
            continue
        out = df[["Close"]].copy().dropna()
        if out.empty:
            continue
        return out
    return None


def last_sma120(series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    if series.empty:
        return None, None
    sma = series.rolling(window=120, min_periods=120).mean()
    last_close = float(series.iloc[-1])
    last_sma = sma.iloc[-1]
    if pd.isna(last_sma) or math.isnan(float(last_sma)):
        return last_close, None
    return last_close, float(last_sma)


def decide_pass(df: pd.DataFrame) -> Tuple[bool, str]:
    close_daily = df["Close"].astype(float)

    close_monthly = close_daily.resample("ME").last().dropna()
    monthly_close, monthly_sma = last_sma120(close_monthly)
    if monthly_sma is not None and monthly_close is not None:
        return monthly_close > monthly_sma, "M"

    close_weekly = close_daily.resample("W-FRI").last().dropna()
    weekly_close, weekly_sma = last_sma120(close_weekly)
    if weekly_sma is not None and weekly_close is not None:
        return weekly_close > weekly_sma, "W"

    daily_close, daily_sma = last_sma120(close_daily)
    if daily_sma is not None and daily_close is not None:
        return daily_close > daily_sma, "D"

    return True, "NO120"


def main() -> int:
    codes = parse_codes(sys.stdin)

    stats = {"M": 0, "W": 0, "D": 0, "NO120": 0, "NO_DATA": 0}
    passed: List[str] = []

    for code in codes:
        df = fetch_ohlcv(code)
        if df is None:
            stats["NO_DATA"] += 1
            continue

        is_passed, route = decide_pass(df)
        if is_passed:
            passed.append(code)
            stats[route] += 1

    for code in passed:
        print(code)

    print(f"PASSED: {len(passed)}")
    print(
        "STATS "
        f"M={stats['M']} "
        f"W={stats['W']} "
        f"D={stats['D']} "
        f"NO120={stats['NO120']} "
        f"NO_DATA={stats['NO_DATA']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
