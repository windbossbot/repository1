from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from datetime import date, datetime, timedelta
from typing import Iterable, List

import pandas as pd
import requests

API_BASE = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService"
API_ENDPOINT = "getStockPriceInfo"
API_URL = f"{API_BASE}/{API_ENDPOINT}"
REQUEST_TIMEOUT = 20
PAGE_SIZE = 1000
MAX_BACKTRACK_DAYS = 20
HISTORY_TRADING_DAYS = 240
MAX_HISTORY_CALENDAR_DAYS = 420

MIN_DAILY_TRADE_VALUE = 3_000_000_000
MIN_MARKET_CAP = 100_000_000_000
MAX_PRICE = 300_000


def to_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    text = str(value).strip().replace(",", "")
    if not text:
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def first_available(row: pd.Series, columns: Iterable[str], default: int = 0) -> int:
    for col in columns:
        if col in row and pd.notna(row[col]):
            return to_int(row[col], default)
    return default


def is_flagged(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    return text not in {"n", "0", "false", "f", "정상", "해당없음", "none"}


class KRXClient:
    def __init__(self, service_key: str) -> None:
        self.service_key = service_key
        self.session = requests.Session()

    def _request_page(self, bas_dt: str, page_no: int) -> dict:
        params = {
            "serviceKey": self.service_key,
            "resultType": "json",
            "numOfRows": str(PAGE_SIZE),
            "pageNo": str(page_no),
            "basDt": bas_dt,
        }
        response = self.session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()

        body = payload.get("response", {}).get("body", {})
        if not isinstance(body, dict):
            return {}
        return body

    def fetch_snapshot(self, bas_dt: str) -> pd.DataFrame:
        body = self._request_page(bas_dt, page_no=1)
        total_count = to_int(body.get("totalCount"), 0)

        items = body.get("items", {}).get("item", [])
        rows: List[dict] = []
        if isinstance(items, dict):
            rows.append(items)
        elif isinstance(items, list):
            rows.extend(items)

        if total_count > len(rows):
            total_pages = (total_count + PAGE_SIZE - 1) // PAGE_SIZE
            for page_no in range(2, total_pages + 1):
                page_body = self._request_page(bas_dt, page_no=page_no)
                page_items = page_body.get("items", {}).get("item", [])
                if isinstance(page_items, dict):
                    rows.append(page_items)
                elif isinstance(page_items, list):
                    rows.extend(page_items)

        return pd.DataFrame(rows)

    def find_latest_snapshot(self) -> tuple[str, pd.DataFrame]:
        today = date.today()
        for i in range(MAX_BACKTRACK_DAYS + 1):
            target = (today - timedelta(days=i)).strftime("%Y%m%d")
            df = self.fetch_snapshot(target)
            if not df.empty:
                return target, df
        raise RuntimeError("No trading snapshot found in recent backtrack window.")

    def fetch_history(self, latest_dt: str) -> pd.DataFrame:
        latest = datetime.strptime(latest_dt, "%Y%m%d").date()
        daily_frames: List[pd.DataFrame] = []
        found_trading_days = 0

        for i in range(MAX_HISTORY_CALENDAR_DAYS + 1):
            target = (latest - timedelta(days=i)).strftime("%Y%m%d")
            df = self.fetch_snapshot(target)
            if df.empty:
                continue
            df = df.copy()
            df["basDt"] = target
            daily_frames.append(df)
            found_trading_days += 1
            if found_trading_days >= HISTORY_TRADING_DAYS:
                break

        if not daily_frames:
            return pd.DataFrame()
        return pd.concat(daily_frames, ignore_index=True)


def enrich_price_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["close"] = out.apply(lambda r: first_available(r, ["clpr", "close", "stckClpr"]), axis=1)
    out["volume"] = out.apply(lambda r: first_available(r, ["trqu", "accTrdVol", "volume"]), axis=1)
    out["trade_value"] = out.apply(lambda r: first_available(r, ["trPrc", "accTrdVal", "tradeValue"]), axis=1)
    out["market_cap"] = out.apply(lambda r: first_available(r, ["mrktTotAmt", "marketCap"]), axis=1)

    missing_trade_value = out["trade_value"] <= 0
    out.loc[missing_trade_value, "trade_value"] = out.loc[missing_trade_value, "close"] * out.loc[missing_trade_value, "volume"]
    return out


def apply_stage1(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return snapshot

    df = enrich_price_fields(snapshot)
    df = df[
        (df["close"] < MAX_PRICE)
        & (df["trade_value"] >= MIN_DAILY_TRADE_VALUE)
        & (df["market_cap"] >= MIN_MARKET_CAP)
    ].copy()

    flag_groups = [
        ["haltYn", "trhtYn", "isTradingHalt"],
        ["mgtIssueYn", "admYn", "isManagementIssue"],
        ["invstCautnYn", "investCautionYn", "isInvestmentCaution"],
    ]

    for cols in flag_groups:
        existing = [c for c in cols if c in df.columns]
        if not existing:
            continue
        flagged_mask = df[existing].apply(lambda col: col.map(is_flagged)).any(axis=1)
        df = df[~flagged_mask]

    return df


def apply_stage2(stage1_df: pd.DataFrame, history_df: pd.DataFrame, latest_dt: str) -> pd.DataFrame:
    if stage1_df.empty or history_df.empty:
        return stage1_df.iloc[0:0].copy()

    history = enrich_price_fields(history_df)

    code_col = "srtnCd" if "srtnCd" in history.columns else "isinCd"
    if code_col not in stage1_df.columns:
        return stage1_df.iloc[0:0].copy()

    history = history.dropna(subset=[code_col]).copy()
    history["basDt"] = pd.to_datetime(history["basDt"], format="%Y%m%d", errors="coerce")
    history = history.sort_values([code_col, "basDt"])

    target_codes = set(stage1_df[code_col].dropna().astype(str).tolist())
    latest_ts = pd.to_datetime(latest_dt, format="%Y%m%d")
    passed_codes: List[str] = []

    for code, group in history.groupby(code_col):
        code_str = str(code)
        if code_str not in target_codes:
            continue

        g = group[group["close"] > 0].copy()
        if g.empty:
            continue

        current_rows = g[g["basDt"] == latest_ts]
        if current_rows.empty:
            continue
        current_close = int(current_rows.iloc[-1]["close"])

        if len(g) < 240:
            continue

        if g.tail(5)["trade_value"].mean() < MIN_DAILY_TRADE_VALUE:
            continue

        ma240 = g.tail(240)["close"].mean()
        if current_close <= ma240:
            continue

        if current_close == int(g["close"].min()):
            continue

        passed_codes.append(code_str)

    out = stage1_df.copy()
    out[code_col] = out[code_col].astype(str)
    out = out[out[code_col].isin(passed_codes)]
    return out


def load_dotenv_if_available() -> None:
    spec = importlib.util.find_spec("dotenv")
    if spec is None:
        return
    module = importlib.import_module("dotenv")
    loader = getattr(module, "load_dotenv", None)
    if callable(loader):
        loader()


def main() -> int:
    load_dotenv_if_available()

    service_key = os.getenv("DATA_GO_KR_SERVICE_KEY", "").strip()
    if not service_key:
        print("ERROR: DATA_GO_KR_SERVICE_KEY is not set.", file=sys.stderr)
        return 1

    client = KRXClient(service_key)

    try:
        latest_dt, latest_snapshot = client.find_latest_snapshot()
        stage1_df = apply_stage1(latest_snapshot)
        history_df = client.fetch_history(latest_dt)
        final_df = apply_stage2(stage1_df, history_df, latest_dt)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    code_col = "srtnCd" if "srtnCd" in final_df.columns else "isinCd"
    if code_col not in final_df.columns:
        print("TOTAL_COUNT=0")
        return 0

    codes = final_df[code_col].dropna().astype(str).tolist()
    if code_col == "srtnCd":
        codes = [code.zfill(6) for code in codes]

    unique_codes = sorted(set(codes))
    for code in unique_codes:
        print(code)
    print(f"TOTAL_COUNT={len(unique_codes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
