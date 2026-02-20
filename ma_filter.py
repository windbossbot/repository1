from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta

import pandas as pd
import FinanceDataReader as fdr

SMA_N = 120
LOOKBACK_YEARS = 15


def sma_last(series: pd.Series, n: int):
    if series is None or len(series) < n:
        return None
    v = series.rolling(n).mean().iloc[-1]
    # NaN이면 "없음"으로 취급 → 다음 단계로 넘김(=없으면 통과 로직의 핵심)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def resample_close(df: pd.DataFrame, rule: str) -> pd.Series:
    return df["Close"].resample(rule).last().dropna()


def decide(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Returns (passed?, used_level)
    used_level: 'M' / 'W' / 'D' / 'NO120'
    """
    close = float(df["Close"].iloc[-1])

    # 1) 120월봉이 "있으면" 여기서 결정 (없으면 다음으로)
    m = resample_close(df, "ME")  # month-end
    m_sma = sma_last(m, SMA_N)
    if m_sma is not None:
        return (close > m_sma, "M")

    # 2) 120주봉이 "있으면" 여기서 결정
    w = resample_close(df, "W-FRI")
    w_sma = sma_last(w, SMA_N)
    if w_sma is not None:
        return (close > w_sma, "W")

    # 3) 120일선이 "있으면" 여기서 결정
    d_sma = sma_last(df["Close"], SMA_N)
    if d_sma is not None:
        return (close > d_sma, "D")

    # 120 자체가 없으면 통과(요청대로)
    return (True, "NO120")


def read_codes_from_stdin() -> list[str]:
    # main.py 출력에서 "6자리 숫자 코드"만 통과
    codes: list[str] = []
    for line in sys.stdin:
        c = line.strip()
        if not c:
            continue
        # TOTAL_COUNT=... 같은 줄 제외
        if c.startswith("TOTAL_COUNT="):
            continue
        # 숫자만 허용
        if not c.isdigit():
            continue
        if len(c) > 6:
            continue
        codes.append(c.zfill(6))
    return codes


def fetch_df(code6: str, start: str) -> pd.DataFrame | None:
    # 환경 따라 KRX: 접두가 필요할 수 있어 2개 시도
    for sym in (code6, f"KRX:{code6}"):
        try:
            df = fdr.DataReader(sym, start)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            continue
    return None


def main() -> int:
    codes = read_codes_from_stdin()
    if not codes:
        print("PASSED: 0")
        return 0

    start = (datetime.today() - timedelta(days=365 * LOOKBACK_YEARS)).strftime("%Y-%m-%d")

    passed_codes: list[str] = []
    stats = {"M": 0, "W": 0, "D": 0, "NO120": 0, "NO_DATA": 0}

    for code in codes:
        df = fetch_df(code, start)
        if df is None:
            stats["NO_DATA"] += 1
            continue

        ok, lvl = decide(df)
        stats[lvl] += 1
        if ok:
            passed_codes.append(code)

    # 통과 종목 출력
    for c in passed_codes:
        print(c)

    # 요약
    print(f"PASSED: {len(passed_codes)}")
    print(f"STATS M={stats['M']} W={stats['W']} D={stats['D']} NO120={stats['NO120']} NO_DATA={stats['NO_DATA']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
