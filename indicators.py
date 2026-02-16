from __future__ import annotations

import pandas as pd


def compute_sma_metrics(df: pd.DataFrame) -> dict[str, float] | None:
    if df.empty or "Close" not in df.columns:
        return None

    close = df["Close"].copy()
    work = pd.DataFrame({"Close": close})
    for window in (20, 60, 120, 240):
        work[f"sma{window}"] = close.rolling(window).mean()

    latest = work.iloc[-1]
    if pd.isna(latest["sma120"]):
        return None

    result = {
        "close": float(latest["Close"]),
        "sma20": float(latest["sma20"]) if pd.notna(latest["sma20"]) else float("nan"),
        "sma60": float(latest["sma60"]) if pd.notna(latest["sma60"]) else float("nan"),
        "sma120": float(latest["sma120"]),
        "sma240": float(latest["sma240"]) if pd.notna(latest["sma240"]) else float("nan"),
        "timestamp": str(work.index[-1]),
    }
    result["distance_to_sma20_pct"] = ((result["close"] / result["sma20"]) - 1.0) * 100 if result["sma20"] == result["sma20"] else float("nan")
    result["distance_to_sma120_pct"] = ((result["close"] / result["sma120"]) - 1.0) * 100
    return result


def is_bull(metrics: dict[str, float], require_sma240: bool = True) -> bool:
    s20 = metrics["sma20"]
    s60 = metrics["sma60"]
    s120 = metrics["sma120"]
    s240 = metrics["sma240"]
    close = metrics["close"]

    if any(v != v for v in (s20, s60, s120)):
        return False
    ordered = s20 > s60 > s120
    if require_sma240 and s240 == s240:
        ordered = ordered and (s120 > s240)
    return ordered and close <= s20 * 1.02


def is_bear(metrics: dict[str, float]) -> bool:
    return metrics["close"] <= metrics["sma120"] * 1.03
