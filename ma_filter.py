import sys
from datetime import datetime, timedelta
import pandas as pd
import FinanceDataReader as fdr

SMA_N = 120

def sma_last(series, n):
    if len(series) < n:
        return None
    return series.rolling(n).mean().iloc[-1]

def resample_close(df, rule):
    return df["Close"].resample(rule).last().dropna()

def decide(df):
    close = df["Close"].iloc[-1]

    m = resample_close(df, "ME")
    m_sma = sma_last(m, SMA_N)
    if m_sma is not None:
        return close > m_sma

    w = resample_close(df, "W-FRI")
    w_sma = sma_last(w, SMA_N)
    if w_sma is not None:
        return close > w_sma

    d_sma = sma_last(df["Close"], SMA_N)
    if d_sma is not None:
        return close > d_sma

    return True

def main():
    codes = []
    with open("codes.txt") as f:
        for line in f:
            codes.append(line.strip())

    start = (datetime.today() - timedelta(days=365*15)).strftime("%Y-%m-%d")

    passed = []

    for code in codes:
        df = fdr.DataReader(code, start)
        if df.empty:
            continue
        if decide(df):
            passed.append(code)

    print("PASSED:", len(passed))

if __name__ == "__main__":
    main()
