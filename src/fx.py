from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import requests

from .storage import load_json, save_json

FX_URL = "https://api.exchangerate.host/latest?base=USD&symbols=KRW"


def _extract_rate(payload: dict) -> float:
    if "rates" in payload and "KRW" in payload["rates"]:
        return float(payload["rates"]["KRW"])
    if "quotes" in payload and "USDKRW" in payload["quotes"]:
        return float(payload["quotes"]["USDKRW"])
    raise ValueError("KRW rate missing in exchangerate.host payload")


def get_usdkrw(cache_path: Path) -> float:
    cached = load_json(cache_path, default={})
    try:
        resp = requests.get(FX_URL, timeout=10)
        resp.raise_for_status()
        rate = _extract_rate(resp.json())
        save_json(
            cache_path,
            {"usdkrw": rate, "updated_utc": datetime.now(timezone.utc).isoformat()},
        )
        return rate
    except Exception:
        if "usdkrw" in cached:
            return float(cached["usdkrw"])
        raise RuntimeError("Failed to fetch USDKRW and no cached FX available")
