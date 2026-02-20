import sys
from datetime import datetime, timedelta

import pandas as pd
import FinanceDataReader as fdr

SMA_N = 120


def sma_last(series: pd.Series, n: int):
    if len(series) < n:
        return None
    return series.rolling(n).mean().iloc[-1]


def resample_close(df: pd.DataFrame, rule: str) -> pd.Series:
    return df["Close"].resample(rule).last().dropna()


def decide(df: pd.DataFrame) -> bool:
    close = float(df["Close"].iloc[-1])

    # 1) 120 month SMA (if exists)
    m = resample_close(df, "ME")
    m_sma = sma_last(m, SMA_N)
    if m_sma is not None:
        return close > float(m_sma)

    # 2) 120 week SMA (if exists)
    w = resample_close(df, "W-FRI")
    w_sma = sma_last(w, SMA_N)
    if w_sma is not None:
        return close > float(w_sma)

    # 3) 120 day SMA (if exists)
    d_sma = sma_last(df["Close"], SMA_N)
    if d_sma is not None:
        return close > float(d_sma)

    # if no 120 data at all -> pass
    return True


def main() -> int:
    # Read codes from STDIN (one per line)
    codes = []
    for line in sys.stdin:
        c = line.strip()
        if c:
            codes.append(c.zfill(6))

    if not codes:
        print("PASSED: 0")
        return 0

    start = (datetime.today() - timedelta(days=365 * 15)).strftime("%Y-%m-%d")

    passed = 0
    for code in codes:
        try:
            df = fdr.DataReader(code, start)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            df = df.copy()
            df.index = pd.to_datetime(df.index)

            if decide(df):
                print(code)
                passed += 1
        except Exception:
            continue

    print(f"PASSED: {passed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
