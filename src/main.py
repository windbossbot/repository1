from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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


def _benchmark_score_us(symbols: list[str]) -> tuple[str, int]:
    data = fetch_daily_ohlcv_batch(symbols, years=5)
    for symbol in symbols:
        frame = data.get(symbol)
        if frame is None or frame.empty:
            continue
        score = compute_score(rebuild_weekly_from_daily_close(frame, source_tz="UTC"))
        if score is not None:
            return symbol, score
    raise RuntimeError(f"No benchmark data for {symbols}")


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

    kospi_benchmark, kospi_score = _benchmark_score_us(["^KS11", "069500.KS"])
    nasdaq_benchmark, nasdaq_score = _benchmark_score_us(["^IXIC"])
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
