from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from typing import Dict, Iterable, Iterator, List

import requests

API_URL = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo"
REQUEST_TIMEOUT = 20
PAGE_SIZE = 1000
MAX_BACKTRACK_DAYS = 20

MIN_PRICE = 2_000
MAX_PRICE = 300_000
MIN_VOLUME = 100_000
MIN_DAILY_TRADE_VALUE = 5_000_000_000


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


def first_available(item: Dict[str, object], candidates: Iterable[str], default: int = 0) -> int:
    for key in candidates:
        if key in item:
            return to_int(item.get(key), default)
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

    def request_page(self, bas_dt: str, page_no: int) -> Dict[str, object]:
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
        return body if isinstance(body, dict) else {}

    def find_latest_bas_dt(self) -> str:
        today = date.today()
        for i in range(MAX_BACKTRACK_DAYS + 1):
            bas_dt = (today - timedelta(days=i)).strftime("%Y%m%d")
            body = self.request_page(bas_dt=bas_dt, page_no=1)
            total_count = to_int(body.get("totalCount"), 0)
            items = body.get("items", {}).get("item", [])
            has_items = isinstance(items, dict) or (isinstance(items, list) and len(items) > 0)
            if total_count > 0 and has_items:
                return bas_dt
        raise RuntimeError("No recent available trading date found.")

    def iter_snapshot_items(self, bas_dt: str) -> Iterator[Dict[str, object]]:
        first = self.request_page(bas_dt=bas_dt, page_no=1)
        total_count = to_int(first.get("totalCount"), 0)
        total_pages = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE)

        yield from self._extract_items(first)

        for page_no in range(2, total_pages + 1):
            body = self.request_page(bas_dt=bas_dt, page_no=page_no)
            yield from self._extract_items(body)

    @staticmethod
    def _extract_items(body: Dict[str, object]) -> Iterator[Dict[str, object]]:
        items = body.get("items", {}).get("item", [])
        if isinstance(items, dict):
            yield items
            return
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    yield item


def passes_stage1(item: Dict[str, object]) -> bool:
    close = first_available(item, ["clpr", "close", "stckClpr"])
    if close < MIN_PRICE or close >= MAX_PRICE:
        return False

    volume = first_available(item, ["trqu", "accTrdVol", "volume"])
    if volume < MIN_VOLUME:
        return False

    trade_value = first_available(item, ["trPrc", "accTrdVal", "tradeValue"])
    if trade_value <= 0:
        trade_value = close * volume
    if trade_value < MIN_DAILY_TRADE_VALUE:
        return False

    # Market cap intentionally skipped when not reliably available from this endpoint.

    flag_groups: List[List[str]] = [
        ["haltYn", "trhtYn", "isTradingHalt"],
        ["mgtIssueYn", "admYn", "isManagementIssue"],
        ["invstCautnYn", "investCautionYn", "isInvestmentCaution"],
    ]
    for group in flag_groups:
        existing = [key for key in group if key in item]
        if not existing:
            continue
        if any(is_flagged(item.get(key)) for key in existing):
            return False

    return True


def main() -> int:
    service_key = os.getenv("DATA_GO_KR_SERVICE_KEY", "").strip()
    if not service_key:
        print("ERROR: DATA_GO_KR_SERVICE_KEY is not set.", file=sys.stderr)
        return 1

    client = KRXClient(service_key)

    try:
        bas_dt = client.find_latest_bas_dt()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    total = 0
    try:
        for item in client.iter_snapshot_items(bas_dt):
            if not passes_stage1(item):
                continue
            code = str(item.get("srtnCd", "")).strip()
            if not code:
                continue
            print(code.zfill(6))
            total += 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"TOTAL_COUNT={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
