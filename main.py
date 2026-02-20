from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

BASE_URL = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo"
REQUEST_TIMEOUT = 20
PAGE_SIZE = 1000
MAX_CALENDAR_BACKTRACK = 20
MAX_HISTORY_DAYS = 300
MAX_HISTORY_CALENDAR_SPAN = 420


def _to_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    text = str(value).strip().replace(",", "")
    if text == "":
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _first_available(row: pd.Series, columns: Iterable[str], default: int = 0) -> int:
    for col in columns:
        if col in row and pd.notna(row[col]):
            return _to_int(row[col], default=default)
    return default


def _flag_truthy(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if text == "":
        return False
    return text not in {"n", "0", "false", "f", "정상", "해당없음", "none"}


class KRXSnapshotFilter:
    def __init__(self, service_key: str) -> None:
        self.service_key = service_key
        self.session = requests.Session()

    def _request(self, bas_dt: str, page_no: int = 1, num_of_rows: int = PAGE_SIZE) -> dict:
        params = {
            "serviceKey": self.service_key,
            "resultType": "json",
            "numOfRows": str(num_of_rows),
            "pageNo": str(page_no),
            "basDt": bas_dt,
        }
        response = self.session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        body = payload.get("response", {}).get("body", {})
        return body if isinstance(body, dict) else {}

    def fetch_snapshot_for_date(self, bas_dt: str) -> pd.DataFrame:
        first_page = self._request(bas_dt=bas_dt, page_no=1)
        total_count = _to_int(first_page.get("totalCount"), 0)
        items = first_page.get("items", {}).get("item", [])

        if isinstance(items, dict):
            all_rows: List[dict] = [items]
        else:
            all_rows = list(items) if items else []

        if total_count <= len(all_rows):
            return pd.DataFrame(all_rows)

        total_pages = (total_count + PAGE_SIZE - 1) // PAGE_SIZE
        for page in range(2, total_pages + 1):
            body = self._request(bas_dt=bas_dt, page_no=page)
            page_items = body.get("items", {}).get("item", [])
            if isinstance(page_items, dict):
                all_rows.append(page_items)
            elif isinstance(page_items, list):
                all_rows.extend(page_items)

        return pd.DataFrame(all_rows)

    def find_latest_trading_date(self) -> tuple[str, pd.DataFrame]:
        today = date.today()
        for step in range(MAX_CALENDAR_BACKTRACK + 1):
            d = today - timedelta(days=step)
            bas_dt = d.strftime("%Y%m%d")
            df = self.fetch_snapshot_for_date(bas_dt)
            if not df.empty:
                return bas_dt, df
        raise RuntimeError("Could not find recent trading date with data.")

    def fetch_historical_snapshots(self, latest_dt: str, needed_days: int = MAX_HISTORY_DAYS) -> pd.DataFrame:
        latest_date = datetime.strptime(latest_dt, "%Y%m%d").date()
        collected: List[pd.DataFrame] = []
        trading_days_found = 0

        for offset in range(MAX_HISTORY_CALENDAR_SPAN + 1):
            d = latest_date - timedelta(days=offset)
            bas_dt = d.strftime("%Y%m%d")
            daily = self.fetch_snapshot_for_date(bas_dt)
            if daily.empty:
                continue
            daily["basDt"] = bas_dt
            collected.append(daily)
            trading_days_found += 1
            if trading_days_found >= needed_days:
                break

        if not collected:
            return pd.DataFrame()
        return pd.concat(collected, ignore_index=True)


def apply_stage1_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    out["close"] = out.apply(lambda r: _first_available(r, ["clpr", "close", "stckClpr"]), axis=1)
    out["volume"] = out.apply(lambda r: _first_available(r, ["trqu", "accTrdVol", "volume"]), axis=1)
    out["trade_value"] = out.apply(lambda r: _first_available(r, ["trPrc", "accTrdVal", "tradeValue"]), axis=1)
    out["market_cap"] = out.apply(lambda r: _first_available(r, ["mrktTotAmt", "marketCap"]), axis=1)

    missing_trade_value = out["trade_value"] <= 0
    out.loc[missing_trade_value, "trade_value"] = out.loc[missing_trade_value, "close"] * out.loc[missing_trade_value, "volume"]

    stage1 = out[
        (out["close"] < 300_000)
        & (out["trade_value"] >= 3_000_000_000)
        & (out["market_cap"] >= 100_000_000_000)
    ].copy()

    halt_cols = ["haltYn", "trhtYn", "isTradingHalt"]
    mgmt_cols = ["mgtIssueYn", "admYn", "isManagementIssue"]
    caution_cols = ["invstCautnYn", "investCautionYn", "isInvestmentCaution"]

    for cols in (halt_cols, mgmt_cols, caution_cols):
        existing = [c for c in cols if c in stage1.columns]
        if not existing:
            continue
        mask = stage1[existing].apply(lambda col: col.map(_flag_truthy)).any(axis=1)
        stage1 = stage1[~mask]

    return stage1


def apply_stage2_filters(latest_filtered: pd.DataFrame, history: pd.DataFrame, latest_dt: str) -> pd.DataFrame:
    if latest_filtered.empty or history.empty:
        return latest_filtered.iloc[0:0].copy()

    history = history.copy()
    history["close"] = history.apply(lambda r: _first_available(r, ["clpr", "close", "stckClpr"]), axis=1)
    history["volume"] = history.apply(lambda r: _first_available(r, ["trqu", "accTrdVol", "volume"]), axis=1)
    history["trade_value"] = history.apply(lambda r: _first_available(r, ["trPrc", "accTrdVal", "tradeValue"]), axis=1)
    missing_trade_value = history["trade_value"] <= 0
    history.loc[missing_trade_value, "trade_value"] = history.loc[missing_trade_value, "close"] * history.loc[missing_trade_value, "volume"]

    code_col = "srtnCd" if "srtnCd" in history.columns else "isinCd"
    if code_col not in latest_filtered.columns:
        return latest_filtered.iloc[0:0].copy()

    history = history.dropna(subset=[code_col]).copy()
    history["basDt"] = pd.to_datetime(history["basDt"], format="%Y%m%d", errors="coerce")
    history = history.sort_values([code_col, "basDt"])

    result_codes = []

    latest_filtered = latest_filtered.dropna(subset=[code_col]).copy()
    latest_filtered = latest_filtered.set_index(code_col, drop=False)

    for code, group in history.groupby(code_col):
        if code not in latest_filtered.index:
            continue

        g = group[group["close"] > 0].copy()
        if g.empty:
            continue

        current = g[g["basDt"] == pd.to_datetime(latest_dt, format="%Y%m%d")]
        if current.empty:
            continue

        current_close = int(current.iloc[-1]["close"])

        recent5 = g.tail(5)
        if len(recent5) < 5:
            continue
        avg_trade_5 = recent5["trade_value"].mean()
        if avg_trade_5 < 3_000_000_000:
            continue

        recent240 = g.tail(240)
        if len(recent240) < 240:
            continue
        ma240 = recent240["close"].mean()
        if current_close <= ma240:
            continue

        if current_close == g["close"].min():
            continue

        result_codes.append(code)

    final_df = latest_filtered.loc[latest_filtered.index.intersection(result_codes)].copy()
    return final_df.reset_index(drop=True)


def main() -> int:
    dotenv_spec = importlib.util.find_spec("dotenv")
    if dotenv_spec is not None:
        dotenv_module = importlib.import_module("dotenv")
        load_dotenv_fn = getattr(dotenv_module, "load_dotenv", None)
        if callable(load_dotenv_fn):
            load_dotenv_fn()
    service_key = os.getenv("DATA_GO_KR_SERVICE_KEY", "").strip()
    if not service_key:
        print("ERROR: DATA_GO_KR_SERVICE_KEY is not set.", file=sys.stderr)
        return 1

    crawler = KRXSnapshotFilter(service_key)

    try:
        latest_dt, latest_snapshot = crawler.find_latest_trading_date()
        stage1 = apply_stage1_filters(latest_snapshot)
        history = crawler.fetch_historical_snapshots(latest_dt=latest_dt, needed_days=MAX_HISTORY_DAYS)
        final_df = apply_stage2_filters(stage1, history, latest_dt)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    code_col = "srtnCd" if "srtnCd" in final_df.columns else "isinCd"
    codes = [str(code).zfill(6) if code_col == "srtnCd" else str(code) for code in final_df[code_col].tolist()]

    for code in codes:
        print(code)
    print(f"TOTAL_COUNT={len(codes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
