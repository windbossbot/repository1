from __future__ import annotations

import sys
from datetime import datetime, timedelta

import pandas as pd
import FinanceDataReader as fdr

SMA_N = 120
LOOKBACK_YEARS = 15

# 디버그: 실패 사유를 stderr로 찍고 싶으면 True
DEBUG = True


def sma_last(series: pd.Series, n: int):
    if series is None or len(series) < n:
        return None
    return series.rolling(n).mean().iloc[-1]


def resample_close(df: pd.DataFrame, rule: str) -> pd.Series:
    # df index must be DatetimeIndex
    return df["Close"].resample(rule).last().dropna()


def decide(df: pd.DataFrame) -> bool:
    close = float(df["Close"].iloc[-1])

    # 1) 120개월선 (가능하면 최우선)
    m = resample_close(df, "ME")  # pandas 최신: "M" 대신 "ME"
    m_sma = sma_last(m, SMA_N)
    if m_sma is not None:
        return close > float(m_sma)

    # 2) 120주봉선 (없으면 주봉)
    w = resample_close(df, "W-FRI")
    w_sma = sma_last(w, SMA_N)
    if w_sma is not None:
        return close > float(w_sma)

    # 3) 120일선 (그것도 없으면 일봉)
    d_sma = sma_last(df["Close"], SMA_N)
    if d_sma is not None:
        return close > float(d_sma)

    # 120 자체가 없으면 통과(요청하신 "없으면 통과" 방식)
    return True


def read_codes_from_stdin() -> list[str]:
    codes: list[str] = []
    for line in sys.stdin:
        c = line.strip()
        if not c:
            continue
        # TOTAL_COUNT= 같은 요약 라인은 무시
        if c.startswith("TOTAL_COUNT="):
            continue
        # 6자리로 정규화
        codes.append(c.zfill(6))
    return codes


def fetch_df_krx(code6: str, start: str) -> pd.DataFrame | None:
    """
    FinanceDataReader가 환경/버전에 따라 입력 포맷에 민감해서
    여러 포맷을 순차로 시도합니다.
    """
    candidates = [
        code6,            # "005930"
        f"KRX:{code6}",   # "KRX:005930" (일부 환경에서 필요)
    ]

    last_err = None
    for sym in candidates:
        try:
            df = fdr.DataReader(sym, start)
            if df is None or df.empty:
                last_err = f"EMPTY({sym})"
                continue
            if "Close" not in df.columns:
                last_err = f"NO_CLOSE({sym})"
                continue
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            last_err = f"{type(e).__name__}({sym}): {str(e)[:160]}"

    if DEBUG and last_err:
        print(f"ERR {code6}: {last_err}", file=sys.stderr)
    return None


def main() -> int:
    codes = read_codes_from_stdin()
    if not codes:
        print("PASSED: 0")
        return 0

    start = (datetime.today() - timedelta(days=365 * LOOKBACK_YEARS)).strftime("%Y-%m-%d")

    passed = 0
    for code in codes:
        df = fetch_df_krx(code, start)
        if df is None:
            continue
        try:
            if decide(df):
                print(code)  # 통과 종목코드 출력
                passed += 1
        except Exception as e:
            if DEBUG:
                print(f"ERR_DECIDE {code}: {type(e).__name__} {str(e)[:160]}", file=sys.stderr)
            continue

    print(f"PASSED: {passed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
