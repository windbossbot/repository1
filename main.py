from __future__ import annotations

import hashlib
import importlib
import json
import math
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"
CACHE_DB = CACHE_DIR / "cache.sqlite3"
CACHE_TTL_SECONDS = 12 * 60 * 60

app = FastAPI(title="Market Regime Analyzer")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


class DataSourceError(Exception):
    pass


# ----------------------------
# Optional dependency helpers
# ----------------------------
def require_module(module_name: str, hint: str | None = None):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        msg = f"Missing dependency: {module_name}"
        if hint:
            msg += f" ({hint})"
        raise DataSourceError(msg) from exc


# ----------------------------
# Cache
# ----------------------------
def ensure_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                expires_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def _cache_key(name: str, params: dict[str, Any] | None = None) -> str:
    params = params or {}
    raw = json.dumps({"name": name, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_get(name: str, params: dict[str, Any] | None = None) -> Any | None:
    key = _cache_key(name, params)
    with sqlite3.connect(CACHE_DB) as conn:
        cur = conn.execute("SELECT value, expires_at FROM cache WHERE key = ?", (key,))
        row = cur.fetchone()
    if not row:
        return None
    value, expires_at = row
    if int(time.time()) > expires_at:
        return None
    return json.loads(value)


def cache_set(name: str, value: Any, params: dict[str, Any] | None = None, ttl: int = CACHE_TTL_SECONDS) -> None:
    key = _cache_key(name, params)
    now = int(time.time())
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute(
            """
            INSERT INTO cache (key, value, expires_at, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                expires_at = excluded.expires_at,
                created_at = excluded.created_at
            """,
            (key, json.dumps(value, default=str), now + ttl, now),
        )
        conn.commit()


# ----------------------------
# Data utils
# ----------------------------
def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.resample("W-FRI")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )


def _to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.resample("ME")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )


def _cached_dataframe(cache_name: str) -> pd.DataFrame | None:
    cached = cache_get(cache_name)
    if cached is None:
        return None
    df = pd.DataFrame(cached)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


def _save_dataframe_cache(cache_name: str, df: pd.DataFrame) -> None:
    out = df.reset_index(names="Date")
    out["Date"] = out["Date"].astype(str)
    cache_set(cache_name, out.to_dict(orient="records"))


# ----------------------------
# Source fetchers
# ----------------------------
def fetch_kospi_daily() -> pd.DataFrame:
    cache_name = "kospi_daily"
    cached_df = _cached_dataframe(cache_name)
    if cached_df is not None:
        return cached_df

    pykrx_stock = require_module("pykrx.stock", "pip install pykrx")
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=365 * 15)).strftime("%Y%m%d")

    try:
        df = pykrx_stock.get_index_ohlcv_by_date(start, end, "1001")
    except Exception as exc:
        raise DataSourceError(f"KOSPI source failure: {exc}") from exc

    if df.empty:
        raise DataSourceError("KOSPI source failure: empty data")

    df = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"})
    df.index = pd.to_datetime(df.index)
    use = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    _save_dataframe_cache(cache_name, use)
    return use


def fetch_nasdaq_daily() -> pd.DataFrame:
    cache_name = "nasdaq_daily"
    cached_df = _cached_dataframe(cache_name)
    if cached_df is not None:
        return cached_df

    pdr = require_module("pandas_datareader.data", "pip install pandas-datareader")
    end = datetime.now()
    start = end - timedelta(days=365 * 15)

    try:
        # stooq 나스닥 100 지수 심볼
        df = pdr.DataReader("^NDQ", "stooq", start=start, end=end)
    except Exception as exc:
        raise DataSourceError(f"NASDAQ source failure: {exc}") from exc

    if df.empty:
        raise DataSourceError("NASDAQ source failure: empty data")

    df = df.sort_index()[["Open", "High", "Low", "Close", "Volume"]]
    _save_dataframe_cache(cache_name, df)
    return df


def fetch_binance_daily(symbol: str) -> pd.DataFrame:
    cache_name = f"binance_{symbol.replace('/', '_').lower()}"
    cached_df = _cached_dataframe(cache_name)
    if cached_df is not None:
        return cached_df

    ccxt = require_module("ccxt", "pip install ccxt")
    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = int((datetime.now(tz=timezone.utc) - timedelta(days=365 * 6)).timestamp() * 1000)

    all_rows: list[list[Any]] = []
    cursor = since_ms
    try:
        while True:
            rows = exchange.fetch_ohlcv(symbol, timeframe="1d", since=cursor, limit=1000)
            if not rows:
                break
            all_rows.extend(rows)
            cursor = rows[-1][0] + 24 * 60 * 60 * 1000
            if len(rows) < 1000:
                break
    except Exception as exc:
        raise DataSourceError(f"{symbol} source failure: {exc}") from exc

    if not all_rows:
        raise DataSourceError(f"{symbol} source failure: empty data")

    df = pd.DataFrame(all_rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
    _save_dataframe_cache(cache_name, df)
    return df


# ----------------------------
# Analysis logic
# ----------------------------
def regime_score_from_weekly(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 30:
        raise DataSourceError("Not enough weekly data")

    c = df.copy()
    c["MA60"] = c["Close"].rolling(60).mean()
    c["MA120"] = c["Close"].rolling(120).mean()

    latest = c.iloc[-1]
    cond1 = bool(latest["Close"] >= latest["MA60"]) if not math.isnan(latest["MA60"]) else False
    cond2 = bool(latest["Close"] >= latest["MA120"]) if not math.isnan(latest["MA120"]) else False

    lookback20 = c.iloc[-21:-1] if len(c) >= 21 else c.iloc[:-1]
    cond3 = bool(latest["Close"] > lookback20["High"].max()) if not lookback20.empty else False

    recent10 = c.iloc[-10:]
    prev10 = c.iloc[-20:-10]
    cond4 = bool(recent10["High"].max() > prev10["High"].max()) if (not recent10.empty and not prev10.empty) else False

    checks = {
        "close_ge_ma60w": cond1,
        "close_ge_ma120w": cond2,
        "close_break_20w_high": cond3,
        "higher_high_recent_10w": cond4,
    }
    return {
        "score": sum(checks.values()),
        "checks": checks,
        "latest_week": str(c.index[-1].date()),
    }


def get_fear_and_greed() -> dict[str, Any]:
    cache_name = "fear_greed"
    cached = cache_get(cache_name)
    if cached is not None:
        return cached

    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        entry = payload["data"][0]
    except Exception as exc:
        raise DataSourceError(f"Fear & Greed source failure: {exc}") from exc

    out = {
        "value": int(entry.get("value", 0)),
        "classification": entry.get("value_classification", "N/A"),
        "timestamp": datetime.fromtimestamp(int(entry.get("timestamp", 0)), tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "source": "alternative.me (Fear & Greed Index)",
    }
    cache_set(cache_name, out)
    return out


# ----------------------------
# Screeners
# ----------------------------
def _calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA60"] = d["Close"].rolling(60).mean()
    d["MA120"] = d["Close"].rolling(120).mean()
    d["Value"] = d["Close"] * d["Volume"]
    return d


def _screen_common_ready(df_daily: pd.DataFrame) -> bool:
    if len(df_daily) < 260:
        return False
    if len(_to_monthly(df_daily)) < 3:
        return False
    if len(_to_weekly(df_daily)) < 130:
        return False
    return True


def _bull_screener_on_df(symbol: str, df_daily: pd.DataFrame, is_stock: bool = False) -> dict[str, Any] | None:
    if not _screen_common_ready(df_daily):
        return None

    d = _calculate_indicators(df_daily)
    w = _to_weekly(df_daily)
    latest = d.iloc[-1]

    avg_val_5d = d["Value"].tail(5).mean()
    if is_stock and avg_val_5d < 10_000_000_000:
        return None

    ma120_prev = d["MA120"].iloc[-6]
    within_5 = abs(latest["Close"] / latest["MA20"] - 1) <= 0.05 if not math.isnan(latest["MA20"]) else False
    overheat = (latest["Close"] / d["High"].tail(252).max() - 1) > 0.40

    cond = [
        w["Close"].iloc[-1] >= w["Close"].rolling(60).mean().iloc[-1],
        w["Close"].iloc[-1] >= w["Close"].rolling(120).mean().iloc[-1],
        latest["Close"] >= latest["MA120"],
        latest["MA120"] > ma120_prev,
        latest["MA20"] >= latest["MA60"],
        within_5,
        not overheat,
    ]
    if not all(bool(x) for x in cond):
        return None

    return {
        "symbol": symbol,
        "price": round(float(latest["Close"]), 4),
        "dist_ma20_pct": round((latest["Close"] / latest["MA20"] - 1) * 100, 2),
        "dist_ma120_pct": round((latest["Close"] / latest["MA120"] - 1) * 100, 2),
        "from_52w_high_pct": round((latest["Close"] / d["High"].tail(252).max() - 1) * 100, 2),
        "avg_value_5d": int(avg_val_5d) if not math.isnan(avg_val_5d) else None,
    }


def _bear_transition_screener_on_df(symbol: str, df_daily: pd.DataFrame, is_stock: bool = False) -> dict[str, Any] | None:
    if not _screen_common_ready(df_daily):
        return None

    d = _calculate_indicators(df_daily)
    w = _to_weekly(df_daily)
    m = _to_monthly(df_daily)
    latest = d.iloc[-1]

    avg_val_5d = d["Value"].tail(5).mean()
    if is_stock and avg_val_5d < 10_000_000_000:
        return None

    ma120m = m["Close"].rolling(120).mean().iloc[-1] if len(m) >= 120 else np.nan
    ma120w = w["Close"].rolling(120).mean().iloc[-1]
    structure_survive = (not math.isnan(ma120m) and latest["Close"] >= ma120m) or (latest["Close"] >= ma120w)

    close = d["Close"]
    ma120 = d["MA120"]
    cross_up = ((close > ma120) & (close.shift(1) <= ma120.shift(1))).tail(60).any()
    now_below = latest["Close"] < latest["MA120"] if not math.isnan(latest["MA120"]) else False
    pos = (latest["Close"] / latest["MA120"] - 1) if not math.isnan(latest["MA120"]) else -999

    recent_peak = d["High"].tail(252).max()
    collapse = (latest["Close"] / recent_peak - 1) <= -0.70

    cond = [structure_survive, bool(cross_up), now_below, (-0.10 <= pos <= 0.02), not collapse]
    if not all(cond):
        return None

    return {
        "symbol": symbol,
        "price": round(float(latest["Close"]), 4),
        "dist_ma20_pct": round((latest["Close"] / latest["MA20"] - 1) * 100, 2),
        "dist_ma120_pct": round((latest["Close"] / latest["MA120"] - 1) * 100, 2),
        "from_52w_high_pct": round((latest["Close"] / recent_peak - 1) * 100, 2),
        "avg_value_5d": int(avg_val_5d) if not math.isnan(avg_val_5d) else None,
    }


def _get_krx_screen_universe(limit: int = 80) -> list[str]:
    pykrx_stock = require_module("pykrx.stock", "pip install pykrx")
    today = datetime.now().strftime("%Y%m%d")
    market = pykrx_stock.get_market_cap_by_ticker(today)
    if market.empty:
        return []
    market = market.sort_values("시가총액", ascending=False)
    tickers = market.index.tolist()[: limit * 2]

    out: list[str] = []
    for t in tickers:
        name = pykrx_stock.get_market_ticker_name(t)
        lname = name.lower()
        if any(x in lname for x in ["etf", "etn", "스팩", "우", "우b", "우선주"]):
            continue
        out.append(t)
        if len(out) >= limit:
            break
    return out


def _fetch_krx_ticker_daily(ticker: str, years: int = 3) -> pd.DataFrame:
    pykrx_stock = require_module("pykrx.stock", "pip install pykrx")
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d")
    df = pykrx_stock.get_market_ohlcv_by_date(start, end, ticker)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"})
    df.index = pd.to_datetime(df.index)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def run_screener(kind: str) -> dict[str, Any]:
    cache_name = "screener"
    params = {"kind": kind}
    cached = cache_get(cache_name, params)
    if cached is not None:
        return cached

    pykrx_stock = None
    try:
        pykrx_stock = require_module("pykrx.stock", "pip install pykrx")
    except DataSourceError:
        pass

    results: list[dict[str, Any]] = []
    failures: list[str] = []

    try:
        for ticker in _get_krx_screen_universe(limit=50):
            df = _fetch_krx_ticker_daily(ticker)
            if df.empty:
                continue
            row = _bull_screener_on_df(ticker, df, True) if kind == "bull" else _bear_transition_screener_on_df(ticker, df, True)
            if row:
                row["asset_type"] = "KRX Stock"
                row["name"] = pykrx_stock.get_market_ticker_name(ticker) if pykrx_stock else ticker
                results.append(row)
    except Exception as exc:
        failures.append(f"KRX screener source failure: {exc}")

    for sym in ["BTC/USDT", "ETH/USDT"]:
        try:
            df = fetch_binance_daily(sym)
            row = _bull_screener_on_df(sym, df) if kind == "bull" else _bear_transition_screener_on_df(sym, df)
            if row:
                row["asset_type"] = "Crypto"
                row["name"] = sym
                results.append(row)
        except Exception as exc:
            failures.append(f"{sym} screener source failure: {exc}")

    out = {
        "kind": kind,
        "count": len(results),
        "rows": sorted(results, key=lambda x: x["symbol"])[:100],
        "failures": failures,
        "sources": ["pykrx (KRX)", "ccxt/binance (Crypto)"],
    }
    cache_set(cache_name, out, params)
    return out


def _demo_analysis_payload() -> dict[str, Any]:
    now = datetime.now().strftime("%Y-%m-%d")
    assets = {
        "KOSPI": {"score": 2, "checks": {"close_ge_ma60w": True, "close_ge_ma120w": False, "close_break_20w_high": False, "higher_high_recent_10w": True}, "latest_week": now, "source": "demo data", "error": "데모 모드"},
        "NASDAQ": {"score": 3, "checks": {"close_ge_ma60w": True, "close_ge_ma120w": True, "close_break_20w_high": False, "higher_high_recent_10w": True}, "latest_week": now, "source": "demo data", "error": "데모 모드"},
        "BTC": {"score": 4, "checks": {"close_ge_ma60w": True, "close_ge_ma120w": True, "close_break_20w_high": True, "higher_high_recent_10w": True}, "latest_week": now, "source": "demo data", "error": "데모 모드"},
        "ETH": {"score": 3, "checks": {"close_ge_ma60w": True, "close_ge_ma120w": True, "close_break_20w_high": True, "higher_high_recent_10w": False}, "latest_week": now, "source": "demo data", "error": "데모 모드"},
    }
    total = sum(v["score"] for v in assets.values())
    regime = "강세장" if total >= 8 else "전환기" if total >= 5 else "약세장"
    return {
        "total_score": total,
        "max_score": 16,
        "regime": regime,
        "latest_week": now,
        "assets": assets,
        "fear_greed": {
            "value": 55,
            "classification": "Neutral",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "alternative.me (Fear & Greed Index) - demo",
            "error": "데모 모드"
        },
        "failures": ["데모 모드: 실제 데이터 소스 대신 샘플 표시"],
        "cache_ttl_hours": CACHE_TTL_SECONDS // 3600,
        "is_demo": True,
    }


# ----------------------------
# FastAPI routes
# ----------------------------
@app.on_event("startup")
def on_startup() -> None:
    ensure_cache()


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "ttl_hours": CACHE_TTL_SECONDS // 3600})


@app.get("/api/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/analyze")
def analyze(demo: bool = False) -> JSONResponse:
    ensure_cache()

    if demo:
        return JSONResponse(_demo_analysis_payload())

    assets = {
        "KOSPI": (fetch_kospi_daily, "pykrx index 1001"),
        "NASDAQ": (fetch_nasdaq_daily, "stooq via pandas-datareader (^NDQ)"),
        "BTC": (lambda: fetch_binance_daily("BTC/USDT"), "ccxt/binance BTC/USDT"),
        "ETH": (lambda: fetch_binance_daily("ETH/USDT"), "ccxt/binance ETH/USDT"),
    }

    per_asset: dict[str, Any] = {}
    failures: list[str] = []
    latest_dates: list[str] = []

    for name, (fetcher, source_name) in assets.items():
        try:
            weekly = _to_weekly(fetcher())
            score = regime_score_from_weekly(weekly)
            score["source"] = source_name
            per_asset[name] = score
            latest_dates.append(score["latest_week"])
        except Exception as exc:
            per_asset[name] = {
                "score": 0,
                "checks": {
                    "close_ge_ma60w": False,
                    "close_ge_ma120w": False,
                    "close_break_20w_high": False,
                    "higher_high_recent_10w": False,
                },
                "latest_week": None,
                "source": source_name,
                "error": str(exc),
            }
            failures.append(f"{name} source failure")

    total = sum(v["score"] for v in per_asset.values())
    regime = "강세장" if total >= 8 else "전환기" if total >= 5 else "약세장"

    try:
        fear_data = get_fear_and_greed()
    except Exception as exc:
        fear_data = {
            "value": None,
            "classification": "N/A",
            "timestamp": None,
            "source": "alternative.me (Fear & Greed Index)",
            "error": str(exc),
        }
        failures.append("Fear & Greed source failure")

    return JSONResponse(
        {
            "total_score": total,
            "max_score": 16,
            "regime": regime,
            "latest_week": max(latest_dates) if latest_dates else None,
            "assets": per_asset,
            "fear_greed": fear_data,
            "failures": failures,
            "cache_ttl_hours": CACHE_TTL_SECONDS // 3600,
            "is_demo": False,
        }
    )


@app.get("/api/screener/bull")
def screener_bull() -> JSONResponse:
    ensure_cache()
    return JSONResponse(run_screener("bull"))


@app.get("/api/screener/bear")
def screener_bear() -> JSONResponse:
    ensure_cache()
    return JSONResponse(run_screener("bear"))
