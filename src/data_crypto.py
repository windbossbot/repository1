from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from .storage import load_json, save_json

COINGECKO_COINS_URL = "https://api.coingecko.com/api/v3/coins/list"
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"


def _get_exchange(exchange_id: str = "binance"):
    import ccxt

    ex_cls = getattr(ccxt, exchange_id)
    return ex_cls({"enableRateLimit": True})


def fetch_crypto_daily_close(base: str, exchange_id: str = "binance", years: int = 4) -> pd.DataFrame:
    ex = _get_exchange(exchange_id)
    symbols = [f"{base}/USDT", f"{base}/BTC", f"{base}/ETH"]
    market_symbol = next((s for s in symbols if s in ex.load_markets()), None)
    if market_symbol is None:
        return pd.DataFrame(columns=["Close"])

    since = int((datetime.now(timezone.utc) - timedelta(days=365 * years)).timestamp() * 1000)
    ohlcv = ex.fetch_ohlcv(market_symbol, timeframe="1d", since=since, limit=1500)
    if not ohlcv:
        return pd.DataFrame(columns=["Close"])

    df = pd.DataFrame(ohlcv, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts")[["Open", "High", "Low", "Close", "Volume"]]


def get_symbol_to_coingecko_id(cache_path: Path) -> dict[str, str]:
    cached = load_json(cache_path, default={})
    if cached:
        return cached
    r = requests.get(COINGECKO_COINS_URL, timeout=20)
    r.raise_for_status()
    mapping: dict[str, str] = {}
    for item in r.json():
        sym = item["symbol"].upper()
        mapping.setdefault(sym, item["id"])
    save_json(cache_path, mapping)
    return mapping


def fetch_crypto_mcap_usd(symbol: str, symbol_map_cache: Path) -> float | None:
    symbol_map = get_symbol_to_coingecko_id(symbol_map_cache)
    coin_id = symbol_map.get(symbol.upper())
    if not coin_id:
        return None
    params = {"vs_currency": "usd", "ids": coin_id, "per_page": 1, "page": 1}
    try:
        r = requests.get(COINGECKO_MARKETS_URL, params=params, timeout=20)
        r.raise_for_status()
        arr = r.json()
        if not arr:
            return None
        mcap = arr[0].get("market_cap")
        return float(mcap) if mcap is not None else None
    except Exception:
        return None
