from __future__ import annotations

import pandas as pd
import requests

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DOW30_URL = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "-")


def _download_html(url: str, timeout: int = 30) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def fetch_sp500_tickers(timeout: int = 30) -> list[str]:
    html = _download_html(SP500_URL, timeout=timeout)
    tables = pd.read_html(html)
    for table in tables:
        if "Symbol" in table.columns:
            return sorted({_normalize_symbol(symbol) for symbol in table["Symbol"].dropna().tolist()})
    raise ValueError("Failed to parse S&P 500 table from Wikipedia")


def fetch_dow30_tickers(timeout: int = 30) -> list[str]:
    html = _download_html(DOW30_URL, timeout=timeout)
    tables = pd.read_html(html)
    for table in tables:
        cols = {str(c).strip().lower(): c for c in table.columns}
        if "symbol" in cols:
            return sorted({_normalize_symbol(symbol) for symbol in table[cols["symbol"]].dropna().tolist()})
    raise ValueError("Failed to parse Dow 30 components table from Wikipedia")
