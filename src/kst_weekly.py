from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

KST = ZoneInfo("Asia/Seoul")


def to_kst_index(df: pd.DataFrame, source_tz: str = "UTC") -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index, utc=False)
    if idx.tz is None:
        idx = idx.tz_localize(source_tz)
    out.index = idx.tz_convert(KST)
    return out.sort_index()


def rebuild_weekly_from_daily_close(df_daily: pd.DataFrame, source_tz: str = "UTC") -> pd.Series:
    if df_daily.empty or "Close" not in df_daily.columns:
        return pd.Series(dtype=float)
    d = to_kst_index(df_daily, source_tz=source_tz)
    return d["Close"].resample("W-SUN", label="right", closed="right").last().dropna()
