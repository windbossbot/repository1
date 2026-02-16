from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Iterable

import pandas as pd
import requests

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"


@dataclass(frozen=True)
class ListedTicker:
    ticker: str
    name: str
    exchange: str
    etf: bool


_EXCHANGE_MAP = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "BATS",
    "V": "IEX",
}


def _download_text(url: str, timeout: int = 30) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def _is_obviously_non_common_stock(name: str) -> bool:
    upper = name.upper()
    blocked_keywords = (
        " PFD",
        "PREFERRED",
        "WARRANT",
        " RIGHT",
        " RIGHTS",
        " UNIT",
        "UNITS",
        " DEPOSITARY",
        "ADR",
        "ETN",
        "TRUST",
        "FUND",
    )
    return any(keyword in upper for keyword in blocked_keywords)


def _normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    return symbol.replace(".", "-")


def parse_nasdaq_listed(text: str) -> list[ListedTicker]:
    lines = [line for line in text.splitlines() if line and not line.startswith("File Creation Time")]
    df = pd.read_csv(StringIO("\n".join(lines)), sep="|")
    out: list[ListedTicker] = []
    for _, row in df.iterrows():
        symbol = str(row.get("Symbol", "")).strip()
        if not symbol or symbol == "nan":
            continue
        out.append(
            ListedTicker(
                ticker=_normalize_symbol(symbol),
                name=str(row.get("Security Name", "")).strip(),
                exchange="NASDAQ",
                etf=str(row.get("ETF", "N")).strip().upper() == "Y",
            )
        )
    return out


def parse_other_listed(text: str) -> list[ListedTicker]:
    lines = [line for line in text.splitlines() if line and not line.startswith("File Creation Time")]
    df = pd.read_csv(StringIO("\n".join(lines)), sep="|")
    out: list[ListedTicker] = []
    for _, row in df.iterrows():
        symbol = str(row.get("ACT Symbol", "")).strip()
        if not symbol or symbol == "nan":
            continue
        exchange_code = str(row.get("Exchange", "")).strip().upper()
        out.append(
            ListedTicker(
                ticker=_normalize_symbol(symbol),
                name=str(row.get("Security Name", "")).strip(),
                exchange=_EXCHANGE_MAP.get(exchange_code, exchange_code or "OTHER"),
                etf=str(row.get("ETF", "N")).strip().upper() == "Y",
            )
        )
    return out


def fetch_nasdaq_listed(timeout: int = 30) -> list[ListedTicker]:
    return parse_nasdaq_listed(_download_text(NASDAQ_LISTED_URL, timeout=timeout))


def fetch_other_listed(timeout: int = 30) -> list[ListedTicker]:
    return parse_other_listed(_download_text(OTHER_LISTED_URL, timeout=timeout))


def filter_listed_tickers(
    tickers: Iterable[ListedTicker],
    include_etf: bool = False,
    light_name_filter: bool = True,
) -> list[ListedTicker]:
    out: list[ListedTicker] = []
    for item in tickers:
        if not include_etf and item.etf:
            continue
        if light_name_filter and _is_obviously_non_common_stock(item.name):
            continue
        out.append(item)
    return out
