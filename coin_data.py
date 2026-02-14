import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
STATE_FILE = Path("state.json")
CANDIDATES_FILE = Path("candidates.json")

DEFAULT_STATE = {
    "last_page": 1,
    "last_index": 0,
    "last_updated_utc": None,
}

STABLE_SYMBOL_KEYWORDS = {
    "usdt",
    "usdc",
    "dai",
    "tusd",
    "fdusd",
    "usde",
    "frax",
    "lusd",
    "pyusd",
    "usdp",
    "gusd",
    "susd",
    "usdn",
    "usdd",
    "eurs",
    "eurt",
    "usdx",
    "usd0",
}
STABLE_NAME_KEYWORDS = {
    "stable",
    "usd",
    "dollar",
    "tether",
    "frax",
    "true usd",
    "binance usd",
    "first digital usd",
    "pax dollar",
    "paypal usd",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_state() -> Dict[str, Any]:
    state = _read_json(STATE_FILE, DEFAULT_STATE.copy())
    if not isinstance(state, dict):
        return DEFAULT_STATE.copy()
    out = DEFAULT_STATE.copy()
    out.update(state)
    return out


def save_state(state: Dict[str, Any]) -> None:
    state = {**DEFAULT_STATE, **state, "last_updated_utc": utc_now_iso()}
    _write_json(STATE_FILE, state)


def reset_state() -> None:
    _write_json(STATE_FILE, DEFAULT_STATE.copy())


def load_candidates() -> List[Dict[str, Any]]:
    data = _read_json(CANDIDATES_FILE, [])
    return data if isinstance(data, list) else []


def save_candidates(items: List[Dict[str, Any]]) -> None:
    _write_json(CANDIDATES_FILE, items)


def reset_candidates() -> None:
    _write_json(CANDIDATES_FILE, [])


@st.cache_data(ttl=900, show_spinner=False)
def fetch_markets_page(page: int, per_page: int = 100, vs_currency: str = "usd") -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """CoinGecko endpoint: /coins/markets"""
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": "false",
        "price_change_percentage": "24h",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data, None
        return [], "시장 데이터 형식이 올바르지 않습니다."
    except requests.RequestException:
        return [], "CoinGecko 시장 데이터를 가져오지 못했습니다. 잠시 후 다시 시도해주세요."


@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily_market_chart(coin_id: str, days: int = 400, vs_currency: str = "usd") -> Tuple[List[List[float]], Optional[str]]:
    """CoinGecko endpoint: /coins/{id}/market_chart"""
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": max(days, 200),
        "interval": "daily",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        prices = payload.get("prices", []) if isinstance(payload, dict) else []
        if isinstance(prices, list):
            return prices, None
        return [], f"{coin_id.upper()} 일봉 데이터 형식이 올바르지 않습니다."
    except requests.RequestException:
        return [], f"{coin_id.upper()} 일봉 데이터를 가져오지 못했습니다."


def is_stablecoin(coin: Dict[str, Any]) -> bool:
    symbol = str(coin.get("symbol", "")).lower()
    name = str(coin.get("name", "")).lower()
    coin_id = str(coin.get("id", "")).lower()

    if symbol in STABLE_SYMBOL_KEYWORDS:
        return True
    text = f"{symbol} {name} {coin_id}"
    if any(k in text for k in STABLE_NAME_KEYWORDS):
        return True
    if symbol.endswith("usd") or symbol.startswith("usd"):
        return True
    return False


def apply_base_filters(coin: Dict[str, Any]) -> bool:
    if is_stablecoin(coin):
        return False
    change = coin.get("price_change_percentage_24h")
    if change is None:
        return False
    if change >= 30:
        return False
    if change <= -50:
        return False
    return True


def build_candidate(coin: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rank": coin.get("market_cap_rank"),
        "id": coin.get("id"),
        "symbol": str(coin.get("symbol", "")).upper(),
        "name": coin.get("name"),
        "market_cap": coin.get("market_cap"),
        "change_24h": coin.get("price_change_percentage_24h"),
        "tags": [],
        "saved_at_utc": utc_now_iso(),
    }
