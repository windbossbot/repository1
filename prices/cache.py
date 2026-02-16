from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


class PriceCache:
    def __init__(self, cache_dir: str | Path = "data/price_cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600

    def _path(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_")
        return self.cache_dir / f"{safe}.csv"

    def get(self, ticker: str) -> pd.DataFrame | None:
        path = self._path(ticker)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.ttl_seconds:
            return None

        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            if df.empty:
                return None
            return df.set_index("Date").sort_index()
        except Exception:
            return None

    def set(self, ticker: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        path = self._path(ticker)
        payload = df.copy().reset_index()
        payload.rename(columns={payload.columns[0]: "Date"}, inplace=True)
        payload.to_csv(path, index=False)
