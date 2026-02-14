from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


def _stock_api():
    from pykrx import stock

    return stock


def _latest_trading_day() -> str:
    stock = _stock_api()
    today = datetime.now().strftime("%Y%m%d")
    for i in range(10):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        tickers = stock.get_market_ticker_list(d, market="KOSPI")
        if tickers:
            return d
    return today


def get_kospi_universe_with_mcap() -> list[dict]:
    stock = _stock_api()
    d = _latest_trading_day()
    tickers = stock.get_market_ticker_list(d, market="KOSPI")
    cap_df = stock.get_market_cap(d, market="KOSPI")
    items = []
    for t in tickers:
        name = stock.get_market_ticker_name(t)
        mcap = None
        if t in cap_df.index:
            mcap = float(cap_df.loc[t, "시가총액"])
        items.append({"symbol": t, "name": name, "mcap_krw": mcap})
    return items


def get_kospi_daily_ohlcv(symbol: str, years: int = 4) -> pd.DataFrame:
    stock = _stock_api()
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d")
    df = stock.get_market_ohlcv_by_date(start, end, symbol)
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    out = pd.DataFrame(
        {
            "Open": df["시가"],
            "High": df["고가"],
            "Low": df["저가"],
            "Close": df["종가"],
            "Volume": df["거래량"],
        }
    )
    out.index = pd.to_datetime(out.index)
    return out
