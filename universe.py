from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from sources.nasdaqtrader import ListedTicker, fetch_nasdaq_listed, fetch_other_listed, filter_listed_tickers
from sources.wikipedia import fetch_dow30_tickers, fetch_sp500_tickers


class UniverseMode(str, Enum):
    ALL = "ALL"
    SP500 = "SP500"
    DOW30 = "DOW30"
    NASDAQ_ONLY = "NASDAQ_ONLY"
    OTHERLISTED_ONLY = "OTHERLISTED_ONLY"


@dataclass(frozen=True)
class TickerRecord:
    ticker: str
    name: str = ""
    exchange: str = ""


def _dedupe(items: list[ListedTicker]) -> list[TickerRecord]:
    seen: dict[str, TickerRecord] = {}
    for item in items:
        if item.ticker not in seen:
            seen[item.ticker] = TickerRecord(ticker=item.ticker, name=item.name, exchange=item.exchange)
    return sorted(seen.values(), key=lambda x: x.ticker)


def _safe_fetch_listed(fetcher, include_etf: bool, light_name_filter: bool) -> list[ListedTicker]:
    logger = logging.getLogger(__name__)
    try:
        return filter_listed_tickers(
            fetcher(),
            include_etf=include_etf,
            light_name_filter=light_name_filter,
        )
    except Exception as exc:
        logger.error("failed to fetch listed universe: %s", exc)
        return []


def build_universe(
    mode: UniverseMode,
    include_etf: bool = False,
    light_name_filter: bool = True,
) -> list[TickerRecord]:
    if mode in (UniverseMode.ALL, UniverseMode.NASDAQ_ONLY, UniverseMode.OTHERLISTED_ONLY):
        nasdaq = _safe_fetch_listed(fetch_nasdaq_listed, include_etf, light_name_filter)
        other = _safe_fetch_listed(fetch_other_listed, include_etf, light_name_filter)

        if mode == UniverseMode.NASDAQ_ONLY:
            return _dedupe(nasdaq)
        if mode == UniverseMode.OTHERLISTED_ONLY:
            return _dedupe(other)
        return _dedupe(nasdaq + other)

    if mode == UniverseMode.SP500:
        try:
            return [TickerRecord(ticker=t) for t in fetch_sp500_tickers()]
        except Exception as exc:
            logging.getLogger(__name__).error("failed to fetch SP500 universe: %s", exc)
            return []

    if mode == UniverseMode.DOW30:
        try:
            return [TickerRecord(ticker=t) for t in fetch_dow30_tickers()]
        except Exception as exc:
            logging.getLogger(__name__).error("failed to fetch DOW30 universe: %s", exc)
            return []

    raise ValueError(f"Unsupported universe mode: {mode}")
