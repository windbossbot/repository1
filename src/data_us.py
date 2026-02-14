from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def _chunks(items: list[str], n: int) -> list[list[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def fetch_daily_ohlcv_batch(symbols: list[str], years: int = 4, chunk_size: int = 200) -> dict[str, pd.DataFrame]:
    if not symbols:
        return {}
    start = (datetime.utcnow() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    out: dict[str, pd.DataFrame] = {}

    for group in _chunks(symbols, chunk_size):
        data = yf.download(
            tickers=group,
            start=start,
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            top = set(data.columns.get_level_values(0))
            for s in group:
                if s not in top:
                    continue
                df = data[s].rename(columns=str.title)
                out[s] = df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]].dropna(how="all")
        elif len(group) == 1:
            df = data.rename(columns=str.title)
            out[group[0]] = df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]].dropna(how="all")
    return out


def fetch_market_cap(symbol: str) -> float | None:
    try:
        tk = yf.Ticker(symbol)
        fi = getattr(tk, "fast_info", None)
        if fi and fi.get("market_cap"):
            return float(fi["market_cap"])
        if fi and fi.get("marketCap"):
            return float(fi["marketCap"])
        info = tk.info
        if info and info.get("marketCap"):
            return float(info["marketCap"])
    except Exception:
        return None
    return None
