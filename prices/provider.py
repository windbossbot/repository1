from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PriceProvider(ABC):
    @abstractmethod
    def get_daily_bars(self, ticker: str) -> pd.DataFrame:
        """Return a DataFrame indexed by datetime with OHLCV columns."""
