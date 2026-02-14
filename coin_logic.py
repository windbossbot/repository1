from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def prices_to_close_series(prices: List[List[float]]) -> pd.Series:
    if not prices:
        return pd.Series(dtype=float)
    df = pd.DataFrame(prices, columns=["ts", "close"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df["close"].dropna().reset_index(drop=True)


def compute_ma_status(prices: List[List[float]]) -> Dict[str, Optional[float]]:
    close_series = prices_to_close_series(prices)
    if close_series.empty:
        return {"close": None, "ma20": None, "ma120": None, "status": "W", "pct_vs_ma120": None}

    close = float(close_series.iloc[-1])
    ma20 = float(close_series.tail(20).mean()) if len(close_series) >= 20 else None
    ma120 = float(close_series.tail(120).mean()) if len(close_series) >= 120 else None

    if ma120 is None:
        status = "W"
        pct_vs_ma120 = None
    else:
        if ma20 is not None and close > ma20 and close > ma120:
            status = "A"
        elif close > ma120:
            status = "O"
        else:
            status = "W"
        pct_vs_ma120 = ((close / ma120) - 1.0) * 100.0

    return {
        "close": close,
        "ma20": ma20,
        "ma120": ma120,
        "status": status,
        "pct_vs_ma120": pct_vs_ma120,
    }


def decide_market_mode(btc_status: str, eth_status: str) -> str:
    if btc_status == "W":
        return "Watch"
    if (btc_status == "A" and eth_status == "A") or (btc_status == "O" and eth_status == "A"):
        return "Aggressive"
    return "Conservative"


def status_to_dot(status: str) -> str:
    if status == "A":
        return "🟢"
    if status == "O":
        return "🟡"
    return "🔴"


def enrich_candidates_with_status(
    candidates: List[Dict[str, Any]],
    market_charts: Dict[str, List[List[float]]],
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for c in candidates:
        cid = c.get("id")
        metrics = compute_ma_status(market_charts.get(cid, []))
        status = metrics["status"]
        enriched.append(
            {
                **c,
                "status": status,
                "dot": status_to_dot(status),
                "pct_vs_ma120": metrics["pct_vs_ma120"],
            }
        )
    return enriched


def sort_for_mode(candidates: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode != "Aggressive":
        return sorted(candidates, key=lambda x: (np.inf if x.get("rank") is None else x.get("rank")))

    priority = {"A": 0, "O": 1, "W": 2}
    return sorted(
        candidates,
        key=lambda x: (
            priority.get(x.get("status", "W"), 2),
            np.inf if x.get("rank") is None else x.get("rank"),
        ),
    )


def format_candidate_line(c: Dict[str, Any]) -> str:
    name = c.get("name", "-")
    symbol = c.get("symbol", "-")
    dot = c.get("dot", "🔴")
    rank = c.get("rank", "-")
    change_24h = c.get("change_24h")
    pct_vs_ma120 = c.get("pct_vs_ma120")

    chg_txt = "N/A" if change_24h is None else f"{change_24h:+.2f}%"
    ma_txt = "N/A" if pct_vs_ma120 is None else f"{pct_vs_ma120:+.2f}%"
    return f"{dot} {name} ({symbol}) | #{rank} | 24h {chg_txt} | MA120대비 {ma_txt}"
