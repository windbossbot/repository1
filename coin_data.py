import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

BINANCE_BASE = "https://api.binance.com/api/v3"
STATE_FILE = Path("state.json")
CANDIDATES_FILE = Path("candidates.json")
BINANCE_HEADERS = {"User-Agent": "Mozilla/5.0"}

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
def fetch_symbols() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    url = f"{BINANCE_BASE}/exchangeInfo"
    try:
        resp = requests.get(url, headers=BINANCE_HEADERS, timeout=10)
        if resp.status_code != 200:
            st.warning(f"Binance /exchangeInfo 응답 코드: {resp.status_code}")
            return [], f"거래심볼 조회 실패 (status: {resp.status_code})"
        payload = resp.json()
        symbols = payload.get("symbols", []) if isinstance(payload, dict) else []
        if not isinstance(symbols, list):
            return [], "거래심볼 데이터 형식이 올바르지 않습니다."

        tradable = [
            s
            for s in symbols
            if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"
        ]
        return tradable, None
    except requests.RequestException:
        return [], "Binance 거래심볼을 가져오지 못했습니다. 잠시 후 다시 시도해주세요."


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ticker_24h_map() -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    url = f"{BINANCE_BASE}/ticker/24hr"
    try:
        resp = requests.get(url, headers=BINANCE_HEADERS, timeout=10)
        if resp.status_code != 200:
            st.warning(f"Binance /ticker/24hr 응답 코드: {resp.status_code}")
            return {}, f"24시간 변동률 조회 실패 (status: {resp.status_code})"
        rows = resp.json()
        if not isinstance(rows, list):
            return {}, "24시간 변동률 데이터 형식이 올바르지 않습니다."
        return {str(r.get("symbol", "")): r for r in rows}, None
    except requests.RequestException:
        return {}, "Binance 24시간 변동률을 가져오지 못했습니다."


@st.cache_data(ttl=900, show_spinner=False)
def fetch_markets_page(page: int, per_page: int = 100, vs_currency: str = "usd") -> Tuple[List[Dict[str, Any]], Optional[str]]:
    symbols, sym_err = fetch_symbols()
    if sym_err:
        return [], sym_err

    ticker_map, ticker_err = fetch_ticker_24h_map()
    if ticker_err:
        st.warning(ticker_err)

    start = max(page - 1, 0) * per_page
    end = start + per_page
    page_symbols = symbols[start:end]
    if not page_symbols:
        return [], None

    out: List[Dict[str, Any]] = []
    for idx, s in enumerate(page_symbols, start=start + 1):
        symbol = str(s.get("symbol", ""))
        ticker = ticker_map.get(symbol, {})
        price_change = ticker.get("priceChangePercent")
        out.append(
            {
                "market_cap_rank": idx,
                "id": symbol,
                "symbol": symbol,
                "name": symbol,
                "market_cap": None,
                "price_change_percentage_24h": float(price_change) if price_change is not None else None,
            }
        )
    return out, None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily_market_chart(coin_id: str, days: int = 400, vs_currency: str = "usd") -> Tuple[List[List[float]], Optional[str]]:
    url = f"{BINANCE_BASE}/klines"
    params = {
        "symbol": str(coin_id).upper(),
        "interval": "1d",
        "limit": 400,
    }
    try:
        resp = requests.get(url, params=params, headers=BINANCE_HEADERS, timeout=10)
        if resp.status_code != 200:
            st.warning(f"Binance /klines/{coin_id} 응답 코드: {resp.status_code}")
            return [], f"{str(coin_id).upper()} 일봉 데이터 조회 실패 (status: {resp.status_code})"
        rows = resp.json()
        if not isinstance(rows, list):
            return [], f"{str(coin_id).upper()} 일봉 데이터 형식이 올바르지 않습니다."

        prices: List[List[float]] = []
        for r in rows:
            if not isinstance(r, list) or len(r) < 5:
                continue
            prices.append([float(r[0]), float(r[4])])
        return prices, None
    except requests.RequestException:
        return [], f"{str(coin_id).upper()} 일봉 데이터를 가져오지 못했습니다."


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
