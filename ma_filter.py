from __future__ import annotations

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
    last_close = float(series.iloc[-1])
    if len(series) < 120:
        return last_close, None
    last_sma = float(series.tail(120).mean())
    return last_close, last_sma


def last_sma60(series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    if series.empty:
        return None, None
    last_close = float(series.iloc[-1])
    if len(series) < 60:
        return last_close, None
    last_sma = float(series.tail(60).mean())
    return last_close, last_sma


def monthly_jump_ratio(close_monthly: pd.Series, offset: int = 0) -> Optional[float]:
    if len(close_monthly) < 2:
        return None
    idx_curr = -1 - offset
    idx_prev = -2 - offset
    if abs(idx_prev) > len(close_monthly):
        return None

    prev_close = float(close_monthly.iloc[idx_prev])
    curr_close = float(close_monthly.iloc[idx_curr])
    if prev_close <= 0:
        return None
    return (curr_close - prev_close) / prev_close


def decide_pass(df: pd.DataFrame) -> Tuple[bool, str]:
    close_daily = df["Close"].astype(float)

    # Analyze monthly first for faster early exclusions.
    close_monthly = close_daily.resample("ME").last().dropna()

    # Additional hard exclusions requested.
    # - Monthly candles below 60: skip.
    # - Latest monthly gain > 40%: skip.
    # - Previous-month gain > 300%: skip.
    if len(close_monthly) < 60:
        return False, "M"
    monthly_jump = monthly_jump_ratio(close_monthly, offset=0)
    if monthly_jump is not None and monthly_jump > 0.40:
        return False, "M"
    prev_monthly_jump = monthly_jump_ratio(close_monthly, offset=1)
    if prev_monthly_jump is not None and prev_monthly_jump > 3.00:
        return False, "M"

    monthly_close, monthly_sma60 = last_sma60(close_monthly)
    monthly_close_120, monthly_sma120 = last_sma120(close_monthly)

    # Exclude when monthly 60MA is below monthly 120MA.
    if monthly_sma60 is not None and monthly_sma120 is not None and monthly_sma60 < monthly_sma120:
        return False, "M"

    if monthly_sma60 is not None and monthly_close is not None and monthly_close < monthly_sma60:
        return False, "M"

    if monthly_sma120 is not None and monthly_close_120 is not None:
        return monthly_close_120 > monthly_sma120, "M"

    close_weekly = close_daily.resample("W-FRI").last().dropna()
    if len(close_weekly) < 60:
        return False, "W"

    weekly_close, weekly_sma60 = last_sma60(close_weekly)
    if weekly_sma60 is not None and weekly_close is not None and weekly_close < weekly_sma60:
        return False, "W"

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
