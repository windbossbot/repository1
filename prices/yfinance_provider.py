from __future__ import annotations

import logging
import random
import time

import pandas as pd
import yfinance as yf

from prices.cache import PriceCache
from prices.provider import PriceProvider


class YFinanceProvider(PriceProvider):
    def __init__(
        self,
        cache: PriceCache,
        max_retries: int = 3,
        base_delay_seconds: float = 1.0,
    ):
        self.cache = cache
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds

    def get_daily_bars(self, ticker: str) -> pd.DataFrame:
        cached = self.cache.get(ticker)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                df = yf.download(
                    tickers=ticker,
                    period="2y",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                if df is None or df.empty:
                    raise ValueError("empty data")

                keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                cleaned = df[keep_cols].dropna(subset=["Close"]).sort_index()
                if cleaned.empty:
                    raise ValueError("empty cleaned data")

                self.cache.set(ticker, cleaned)
                return cleaned
            except Exception as exc:
                last_error = exc
                wait = self.base_delay_seconds * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
                logging.getLogger(__name__).warning(
                    "yfinance failed for %s (attempt %s/%s): %s", ticker, attempt, self.max_retries, exc
                )
                time.sleep(wait)

        raise RuntimeError(f"failed to download {ticker}") from last_error
