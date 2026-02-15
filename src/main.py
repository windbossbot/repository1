from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from .data_crypto import fetch_crypto_daily_close, fetch_crypto_mcap_usd
from .data_us import fetch_daily_ohlcv_batch, fetch_market_cap
from .fx import get_usdkrw
from .kst_weekly import rebuild_weekly_from_daily_close
from .pager import GroupInfo, build_page, compute_allocation, rank_groups
from .score import compute_score
from .storage import load_json, save_json
from .universe_crypto import get_crypto_bases
from .universe_kospi import get_kospi_daily_ohlcv, get_kospi_universe_with_mcap
from .universe_nasdaq import fetch_nasdaq_symbols

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FX_CACHE = DATA / "cache_fx.json"
NASDAQ_CACHE = DATA / "cache_symbols_nasdaq.txt"
COINGECKO_SYMBOL_CACHE = DATA / "cache_coingecko_symbols.json"
STATE_PATH = DATA / "state.json"
KST = ZoneInfo("Asia/Seoul")


def _state_default() -> dict:
    return {
        "page": 0,
        "cursors": {
            "KOSPI": {"bull": 0, "bear": 0},
            "NASDAQ": {"bull": 0, "bear": 0},
            "CRYPTO": {"bull": 0, "bear": 0},
        },
    }


def _benchmark_cache_path(group: str) -> Path:
    return DATA / f"cache_benchmark_{group.upper()}.json"


def _serialize_daily_close(df: pd.DataFrame) -> list[dict]:
    out = []
    for ts, row in df.sort_index().iterrows():
        c = row.get("Close")
        if pd.isna(c):
            continue
        out.append({"ts": pd.Timestamp(ts).isoformat(), "close": float(c)})
    return out


def _deserialize_daily_close(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["Close"])
    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=False)
    frame = frame.set_index("ts").sort_index()
    return pd.DataFrame({"Close": pd.to_numeric(frame["close"], errors="coerce")}).dropna()


def _fetch_yf_daily_with_retry(symbol: str, min_days: int = 900) -> pd.DataFrame:
    start = (datetime.utcnow() - timedelta(days=365 * 6)).strftime("%Y-%m-%d")
    delays = [1, 2, 4]
    last_exc: Exception | None = None
    for i in range(len(delays)):
        try:
            data = yf.download(
                tickers=symbol,
                start=start,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="ticker",
            )
            if isinstance(data.columns, pd.MultiIndex):
                if symbol in data.columns.get_level_values(0):
                    data = data[symbol]
            data = data.rename(columns=str.title)
            if "Close" not in data.columns:
                raise ValueError("missing Close column")
            data = data[["Close"]].dropna()
            if len(data) < min_days:
                raise ValueError(f"insufficient rows: {len(data)} < {min_days}")
            return data
        except Exception as exc:
            last_exc = exc
            time.sleep(delays[i])
    if last_exc:
        raise last_exc
    raise RuntimeError(f"failed to fetch {symbol}")


def _load_benchmark_cache(group: str) -> tuple[pd.DataFrame, str] | tuple[None, None]:
    cached = load_json(_benchmark_cache_path(group), default={})
    rows = cached.get("rows") if isinstance(cached, dict) else None
    if not rows:
        return None, None
    df = _deserialize_daily_close(rows)
    if df.empty:
        return None, None
    return df, str(cached.get("symbol", "CACHE"))


def _save_benchmark_cache(group: str, symbol: str, daily_close: pd.DataFrame) -> None:
    save_json(
        _benchmark_cache_path(group),
        {
            "group": group,
            "symbol": symbol,
            "asof_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
            "rows": _serialize_daily_close(daily_close),
        },
    )


def _benchmark_score_us(symbols: list[str], group: str) -> tuple[str, int]:
    attempted = []
    last_exc: Exception | None = None
    cache_exists = _benchmark_cache_path(group).exists()

    use_symbols = list(symbols)
    if group.upper() == "KOSPI":
        use_symbols = ["^KS11", "KOSPI.KS", "069500.KS"]

    for symbol in use_symbols:
        attempted.append(symbol)
        try:
            daily = _fetch_yf_daily_with_retry(symbol, min_days=900)
            score = compute_score(rebuild_weekly_from_daily_close(daily, source_tz="UTC"))
            if score is not None:
                _save_benchmark_cache(group, symbol, daily)
                return symbol, score
            last_exc = RuntimeError("score unavailable due to insufficient weekly history")
        except Exception as exc:
            last_exc = exc

    cached_df, cached_symbol = _load_benchmark_cache(group)
    if cached_df is not None:
        score = compute_score(rebuild_weekly_from_daily_close(cached_df, source_tz="UTC"))
        if score is not None:
            return cached_symbol or "CACHE", score

    if group.upper() == "KOSPI":
        try:
            from pykrx import stock

            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=365 * 6)).strftime("%Y%m%d")
            idx_df = stock.get_index_ohlcv_by_date(start, end, "1001")
            if not idx_df.empty and "종가" in idx_df.columns:
                proxy = pd.DataFrame({"Close": pd.to_numeric(idx_df["종가"], errors="coerce")}).dropna()
                proxy.index = pd.to_datetime(proxy.index)
                score = compute_score(rebuild_weekly_from_daily_close(proxy, source_tz="Asia/Seoul"))
                if score is not None:
                    _save_benchmark_cache(group, "KOSPI_INDEX_1001", proxy)
                    return "KOSPI_INDEX_1001", score
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        f"Benchmark fetch failed for group={group}; attempted_tickers={attempted}; "
        f"cache_exists={cache_exists}; last_exception={repr(last_exc)}; "
        "yfinance blocked/unavailable; try again later or use cached benchmark"
    )


def _crypto_benchmark_score() -> int:
    btc = fetch_crypto_daily_close("BTC", years=5)
    eth = fetch_crypto_daily_close("ETH", years=5)
    btc_score = compute_score(rebuild_weekly_from_daily_close(btc, source_tz="UTC"))
    eth_score = compute_score(rebuild_weekly_from_daily_close(eth, source_tz="UTC"))
    if btc_score is None or eth_score is None:
        raise RuntimeError("Failed to compute BTC/ETH benchmark scores")
    return round((btc_score + eth_score) / 2)


def _state_from_score(score: int) -> str:
    if score >= 3:
        return "BULL"
    if score <= 1:
        return "BEAR"
    return "TURN"


def _prepare_group_lists(usdkrw: float) -> dict:
    lists = {
        "KOSPI": {"bull": [], "bear": []},
        "NASDAQ": {"bull": [], "bear": []},
        "CRYPTO": {"bull": [], "bear": []},
    }

    for item in get_kospi_universe_with_mcap():
        if item.get("mcap_krw") is None:
            continue
        daily = get_kospi_daily_ohlcv(item["symbol"], years=5)
        if daily.empty or "Close" not in daily.columns:
            continue
        score = compute_score(rebuild_weekly_from_daily_close(daily, source_tz="Asia/Seoul"))
        if score is None:
            continue
        price_krw = float(daily["Close"].dropna().iloc[-1])
        if price_krw >= 500000:
            continue
        row = {
            "group": "KOSPI",
            "symbol": item["symbol"],
            "name": item["name"],
            "price_krw": price_krw,
            "mcap_krw": float(item["mcap_krw"]),
            "score": score,
            "bucket": "BULL" if score >= 3 else "BEAR",
        }
        if score >= 3:
            lists["KOSPI"]["bull"].append(row)
        elif score <= 1:
            lists["KOSPI"]["bear"].append(row)

    us_universe = fetch_nasdaq_symbols(NASDAQ_CACHE)
    us_symbols = [item["symbol"] for item in us_universe]
    daily_map = fetch_daily_ohlcv_batch(us_symbols, years=5)
    name_map = {item["symbol"]: item["name"] for item in us_universe}
    for symbol, daily in daily_map.items():
        if daily.empty or "Close" not in daily.columns:
            continue
        mcap_usd = fetch_market_cap(symbol)
        if mcap_usd is None:
            continue
        score = compute_score(rebuild_weekly_from_daily_close(daily, source_tz="UTC"))
        if score is None:
            continue
        price_krw = float(daily["Close"].dropna().iloc[-1]) * usdkrw
        if price_krw >= 500000:
            continue
        row = {
            "group": "NASDAQ",
            "symbol": symbol,
            "name": name_map.get(symbol, symbol),
            "price_krw": price_krw,
            "mcap_krw": mcap_usd * usdkrw,
            "score": score,
            "bucket": "BULL" if score >= 3 else "BEAR",
        }
        if score >= 3:
            lists["NASDAQ"]["bull"].append(row)
        elif score <= 1:
            lists["NASDAQ"]["bear"].append(row)

    for coin in get_crypto_bases("binance"):
        mcap_usd = fetch_crypto_mcap_usd(coin["symbol"], COINGECKO_SYMBOL_CACHE)
        if mcap_usd is None:
            continue
        daily = fetch_crypto_daily_close(coin["symbol"], years=5)
        if daily.empty or "Close" not in daily.columns:
            continue
        score = compute_score(rebuild_weekly_from_daily_close(daily, source_tz="UTC"))
        if score is None:
            continue
        price_krw = float(daily["Close"].dropna().iloc[-1]) * usdkrw
        if price_krw >= 500000:
            continue
        row = {
            "group": "CRYPTO",
            "symbol": coin["symbol"],
            "name": coin["name"],
            "price_krw": price_krw,
            "mcap_krw": mcap_usd * usdkrw,
            "score": score,
            "bucket": "BULL" if score >= 3 else "BEAR",
        }
        if score >= 3:
            lists["CRYPTO"]["bull"].append(row)
        elif score <= 1:
            lists["CRYPTO"]["bear"].append(row)

    for group in lists:
        lists[group]["bull"].sort(key=lambda x: x["mcap_krw"], reverse=True)
        lists[group]["bear"].sort(key=lambda x: x["mcap_krw"], reverse=True)
    return lists


def generate_page(page_num: int, state: dict) -> dict:
    usdkrw = get_usdkrw(FX_CACHE)

    kospi_benchmark, kospi_score = _benchmark_score_us(["^KS11", "KOSPI.KS", "069500.KS"], group="KOSPI")
    nasdaq_benchmark, nasdaq_score = _benchmark_score_us(["^IXIC"], group="NASDAQ")
    crypto_score = _crypto_benchmark_score()

    groups = [
        GroupInfo("KOSPI", kospi_score, _state_from_score(kospi_score), kospi_benchmark),
        GroupInfo("NASDAQ", nasdaq_score, _state_from_score(nasdaq_score), nasdaq_benchmark),
        GroupInfo("CRYPTO", crypto_score, _state_from_score(crypto_score), "BTC+ETH"),
    ]
    ranked = rank_groups(groups)
    allocation = compute_allocation(ranked)

    lists = _prepare_group_lists(usdkrw)
    items, new_cursors = build_page(ranked, allocation, lists, state["cursors"])

    result = {
        "asof_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
        "fx_usdkrw": round(usdkrw, 4),
        "group_ranking": [
            {"group": g.group, "benchmark": g.benchmark, "score": g.score, "state": g.state}
            for g in ranked
        ],
        "allocation": allocation,
        "items": items,
        "cursors": new_cursors,
    }

    state["page"] = page_num
    state["cursors"] = new_cursors
    save_json(STATE_PATH, state)
    return result


def cmd_reset() -> None:
    state = _state_default()
    save_json(STATE_PATH, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def cmd_page(page: int) -> None:
    state = load_json(STATE_PATH, _state_default())
    if page <= 1:
        state = _state_default()
    result = generate_page(page, state)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_next() -> None:
    state = load_json(STATE_PATH, _state_default())
    next_page = int(state.get("page", 0)) + 1
    result = generate_page(next_page, state)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-key KRW-normalized screener CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    page_parser = sub.add_parser("page")
    page_parser.add_argument("--page", type=int, required=True)
    sub.add_parser("next")
    sub.add_parser("reset")

    args = parser.parse_args()
    if args.cmd == "page":
        cmd_page(args.page)
    elif args.cmd == "next":
        cmd_next()
    elif args.cmd == "reset":
        cmd_reset()


if __name__ == "__main__":
    main()
