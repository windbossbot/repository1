from __future__ import annotations

import hashlib
import os
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "cache.sqlite3"
TTL_SECONDS = 12 * 60 * 60

app = FastAPI(title="주식 코인 필터기")
templates = Jinja2Templates(directory="templates")

MARKETS = {
    "KOSPI": {"kind": "stooq", "symbol": "^kospi", "asset": "지수"},
    "NASDAQ": {"kind": "stooq", "symbol": "^ndq", "asset": "지수"},
    "BTC/USD": {"kind": "kraken", "symbol": "BTC/USD", "asset": "코인"},
    "ETH/USD": {"kind": "kraken", "symbol": "ETH/USD", "asset": "코인"},
}

NASDAQ_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "AMD", "NFLX", "INTC",
]


@dataclass
class FxResult:
    rate: float
    source: str
    fallback_used: bool


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_cache (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                expires_at INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def _key(name: str, params: dict[str, Any] | None = None) -> str:
    raw = json.dumps({"name": name, "params": params or {}}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_get(name: str, params: dict[str, Any] | None = None) -> Any | None:
    k = _key(name, params)
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT payload, expires_at FROM api_cache WHERE cache_key = ?", (k,)
        ).fetchone()
    if not row:
        return None
    payload, expires_at = row
    if int(time.time()) > expires_at:
        return None
    return json.loads(payload)


def cache_set(name: str, payload: Any, params: dict[str, Any] | None = None, ttl: int = TTL_SECONDS) -> None:
    k = _key(name, params)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO api_cache(cache_key, payload, expires_at)
            VALUES (?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                payload = excluded.payload,
                expires_at = excluded.expires_at
            """,
            (k, json.dumps(payload), int(time.time()) + ttl),
        )
        conn.commit()


def fetch_stooq_weekly(symbol: str) -> pd.DataFrame:
    cached = cache_get("stooq_weekly", {"symbol": symbol})
    if cached:
        return pd.DataFrame(cached).assign(Date=lambda d: pd.to_datetime(d["Date"]))\
            .set_index("Date").sort_index()

    url = f"https://stooq.com/q/d/l/?s={symbol}&i=w"
    df = pd.read_csv(url)
    if df.empty or "Date" not in df.columns:
        raise ValueError(f"stooq 데이터 없음: {symbol}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    cache_set("stooq_weekly", df.reset_index().assign(Date=lambda d: d["Date"].astype(str)).to_dict("records"), {"symbol": symbol})
    return df


def fetch_kraken_weekly(symbol: str) -> pd.DataFrame:
    cached = cache_get("kraken_weekly", {"symbol": symbol})
    if cached:
        return pd.DataFrame(cached).assign(Date=lambda d: pd.to_datetime(d["Date"]))\
            .set_index("Date").sort_index()

    exchange = ccxt.kraken({"enableRateLimit": True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1w", limit=260)
    if not ohlcv:
        raise ValueError(f"kraken 데이터 없음: {symbol}")

    df = pd.DataFrame(ohlcv, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].sort_index()

    cache_set("kraken_weekly", df.reset_index().assign(Date=lambda d: d["Date"].astype(str)).to_dict("records"), {"symbol": symbol})
    return df


def score_weekly_regime(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 60:
        raise ValueError("데이터가 부족합니다")

    close = df["Close"]
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    rsi = _rsi(close, 14)

    latest = close.index[-1]
    checks = {
        "close_ge_ma20": bool(close.iloc[-1] >= ma20.iloc[-1]),
        "ma20_ge_ma50": bool(ma20.iloc[-1] >= ma50.iloc[-1]),
        "rsi_ge_50": bool(rsi.iloc[-1] >= 50),
        "mom_13w": bool(close.iloc[-1] >= close.iloc[-14]),
    }
    score = sum(int(v) for v in checks.values())
    if score >= 3:
        label = "강세"
    elif score == 2:
        label = "중립"
    else:
        label = "약세"

    return {
        "evaluation": label,
        "score": score,
        "latest_week": latest.strftime("%Y-%m-%d"),
        "checks": checks,
    }


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, 1e-12)
    return 100 - (100 / (1 + rs))


def get_usdkrw() -> FxResult:
    cached = cache_get("usdkrw")
    if cached:
        return FxResult(rate=float(cached["rate"]), source=cached["source"], fallback_used=bool(cached["fallback_used"]))

    urls = [
        ("open.er-api", "https://open.er-api.com/v6/latest/USD"),
        ("exchangerate.host", "https://api.exchangerate.host/latest?base=USD&symbols=KRW"),
    ]
    for source, url in urls:
        try:
            res = requests.get(url, timeout=8)
            res.raise_for_status()
            data = res.json()
            rates = data.get("rates", {})
            rate = float(rates["KRW"])
            out = {"rate": rate, "source": source, "fallback_used": False}
            cache_set("usdkrw", out)
            return FxResult(**out)
        except Exception:
            continue

    out = {"rate": 1350.0, "source": "fallback", "fallback_used": True}
    cache_set("usdkrw", out)
    return FxResult(**out)


def fetch_stock_weekly(symbol: str) -> pd.DataFrame:
    # stooq 미주 종목 형식: aapl.us
    return fetch_stooq_weekly(f"{symbol.lower()}.us")


def bull_route(score: int) -> bool:
    return score >= 3


def bear_route(score: int) -> bool:
    return score <= 1


def run_screener(kind: str) -> dict[str, Any]:
    fx = get_usdkrw()
    selected = bull_route if kind == "bull" else bear_route
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    excluded_by_price_cap = 0

    for ticker in NASDAQ_TICKERS:
        try:
            df = fetch_stock_weekly(ticker)
            result = score_weekly_regime(df)
            usd_price = float(df["Close"].iloc[-1])
            krw_price = usd_price * fx.rate
            if krw_price > 500_000:
                excluded_by_price_cap += 1
                continue
            if selected(result["score"]):
                rows.append(
                    {
                        "ticker": ticker,
                        "evaluation": result["evaluation"],
                        "score": f"{result['score']}/4",
                        "latest_week": result["latest_week"],
                        "price_usd": round(usd_price, 2),
                        "price_krw": int(krw_price),
                    }
                )
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    return {
        "kind": kind,
        "rows": rows,
        "fx": {"rate": fx.rate, "source": fx.source, "fallback_used": fx.fallback_used},
        "excluded_by_price_cap": excluded_by_price_cap,
        "errors": errors,
    }


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/analyze")
def analyze() -> JSONResponse:
    rows = []
    for asset_name, info in MARKETS.items():
        try:
            df = fetch_stooq_weekly(info["symbol"]) if info["kind"] == "stooq" else fetch_kraken_weekly(info["symbol"])
            result = score_weekly_regime(df)
            rows.append(
                {
                    "evaluation": result["evaluation"],
                    "asset": asset_name,
                    "score": f"{result['score']}/4",
                    "error": "",
                    "latest_week": result["latest_week"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "evaluation": "오류",
                    "asset": asset_name,
                    "score": "0/4",
                    "error": str(exc),
                    "latest_week": "-",
                }
            )
    return JSONResponse({"rows": rows})


@app.get("/api/screener/bull")
def screener_bull() -> JSONResponse:
    return JSONResponse(run_screener("bull"))


@app.get("/api/screener/bear")
def screener_bear() -> JSONResponse:
    return JSONResponse(run_screener("bear"))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT") or "8000")
    host = "0.0.0.0"
    print(f"Local URL: http://127.0.0.1:{port}")
    uvicorn.run("main:app", host=host, port=port)
