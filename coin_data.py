import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

CMC_BASE = "https://pro-api.coinmarketcap.com"
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


def get_cmc_headers() -> Optional[Dict[str, str]]:
    key = os.getenv("CMC_API_KEY")
    if not key:
        st.warning("CMC_API_KEY 환경변수가 없습니다. Render 환경변수에 API 키를 설정해주세요.")
        return None
    return {
        "X-CMC_PRO_API_KEY": key,
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    }


def has_cmc_api_key() -> bool:
    return bool(os.getenv("CMC_API_KEY"))


def _cmc_get(endpoint: str, params: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    headers = get_cmc_headers()
    if headers is None:
        return None, "CMC API 키가 없어 데이터를 조회할 수 없습니다."

    url = f"{CMC_BASE}{endpoint}"
    retries = 3
    for i in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 429 and i < retries - 1:
                time.sleep(1.5 * (2**i))
                continue
            if resp.status_code != 200:
                st.warning(f"CMC {endpoint} 응답 코드: {resp.status_code}")
                return None, f"CMC {endpoint} 조회 실패 (status: {resp.status_code})"
            payload = resp.json()
            if not isinstance(payload, dict):
                return None, f"CMC {endpoint} 응답 형식이 올바르지 않습니다."
            return payload, None
        except requests.RequestException:
            if i < retries - 1:
                time.sleep(1.5 * (2**i))
                continue
            return None, f"CMC {endpoint} 요청 중 네트워크 오류가 발생했습니다."
    return None, f"CMC {endpoint} 조회에 실패했습니다."


@st.cache_data(ttl=900, show_spinner=False)
def fetch_markets_page(page: int, per_page: int = 100, vs_currency: str = "usd") -> Tuple[List[Dict[str, Any]], Optional[str]]:
    endpoint = "/v1/cryptocurrency/listings/latest"
    start = 1 + (max(page, 1) - 1) * per_page
    params = {
        "start": start,
        "limit": per_page,
        "convert": "USD",
    }
    payload, err = _cmc_get(endpoint, params)
    if err:
        return [], err

    data = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(data, list):
        return [], "시장 데이터 형식이 올바르지 않습니다."

    out: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        quote = row.get("quote", {}).get("USD", {}) if isinstance(row.get("quote"), dict) else {}
        out.append(
            {
                "market_cap_rank": row.get("cmc_rank"),
                "id": row.get("id"),
                "symbol": str(row.get("symbol", "")),
                "name": row.get("name"),
                "market_cap": quote.get("market_cap"),
                "price_change_percentage_24h": quote.get("percent_change_24h"),
            }
        )
    return out, None


def _extract_quotes_from_ohlcv_data(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        if "quotes" in data and isinstance(data["quotes"], list):
            return [q for q in data["quotes"] if isinstance(q, dict)]
        quotes: List[Dict[str, Any]] = []
        for v in data.values():
            quotes.extend(_extract_quotes_from_ohlcv_data(v))
        return quotes
    if isinstance(data, list):
        quotes: List[Dict[str, Any]] = []
        for v in data:
            quotes.extend(_extract_quotes_from_ohlcv_data(v))
        return quotes
    return []


@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily_market_chart(coin_id: str, days: int = 400, vs_currency: str = "usd") -> Tuple[List[List[float]], Optional[str]]:
    endpoint = "/v2/cryptocurrency/ohlcv/historical"
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=max(200, min(days, 400)) + 10)
    params: Dict[str, Any] = {
        "time_start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interval": "daily",
        "count": max(200, min(days, 400)),
        "convert": "USD",
    }

    coin_text = str(coin_id)
    if coin_text.isdigit():
        params["id"] = coin_text
    else:
        params["symbol"] = coin_text.upper()

    payload, err = _cmc_get(endpoint, params)
    if err:
        return [], err

    data = payload.get("data") if isinstance(payload, dict) else None
    quotes = _extract_quotes_from_ohlcv_data(data)
    if not quotes:
        st.warning(f"CMC {endpoint} 데이터 없음: {coin_id}")
        return [], f"CMC {endpoint}에서 {coin_id} 일봉 데이터를 받지 못했습니다."

    prices: List[List[float]] = []
    for q in quotes:
        t = q.get("timestamp")
        close = q.get("quote", {}).get("USD", {}).get("close")
        if t is None or close is None:
            continue
        try:
            dt = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
            ts_ms = int(dt.timestamp() * 1000)
            prices.append([float(ts_ms), float(close)])
        except Exception:
            continue

    prices.sort(key=lambda x: x[0])
    return prices[-400:], None


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
