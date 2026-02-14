from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"


def fetch_nasdaq_symbols(cache_path: Path) -> list[dict]:
    text = None
    try:
        r = requests.get(NASDAQ_LISTED_URL, timeout=20)
        r.raise_for_status()
        text = r.text
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
    except Exception:
        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
        else:
            raise RuntimeError("Unable to fetch NASDAQ listed file and no cache available")

    lines = [ln for ln in text.splitlines() if "File Creation Time" not in ln and ln.strip()]
    df = pd.DataFrame([row.split("|") for row in lines[1:]], columns=lines[0].split("|"))

    if "Test Issue" in df.columns:
        df = df[df["Test Issue"] != "Y"]
    if "ETF" in df.columns:
        df = df[df["ETF"] != "Y"]

    df = df[df["Symbol"].str.len() > 0]
    return [
        {"symbol": s.strip(), "name": n.strip()}
        for s, n in zip(df["Symbol"].tolist(), df["Security Name"].tolist())
    ]
