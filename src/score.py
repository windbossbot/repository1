from __future__ import annotations

import pandas as pd


def compute_score(close_w: pd.Series) -> int | None:
    s = close_w.dropna().astype(float)
    if len(s) < 130:
        return None
    c = s.iloc[-1]
    sma60 = s.rolling(60).mean().iloc[-1]
    sma120 = s.rolling(120).mean().iloc[-1]
    h20 = s.iloc[-20:].max()
    h10 = s.iloc[-10:].max()
    checks = [c > sma60, c > sma120, c >= h20, c >= h10]
    return int(sum(bool(x) for x in checks))
