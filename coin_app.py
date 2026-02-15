from typing import Dict, List

import streamlit as st

from coin_data import (
    apply_base_filters,
    build_candidate,
    fetch_daily_market_chart,
    fetch_markets_page,
    load_candidates,
    load_state,
    reset_candidates,
    reset_state,
    save_candidates,
    save_state,
    utc_now_iso,
)
from coin_logic import (
    compute_ma_status,
    decide_market_mode,
    enrich_candidates_with_status,
    format_candidate_line,
    sort_for_mode,
)

st.set_page_config(page_title="COIN Screener v1", layout="wide")
st.title("COIN Screener v1")

MAX_CANDIDATES = 10


def run_scan(max_scan_batches: int) -> None:
    state = load_state()
    candidates = load_candidates()
    existing_ids = {c.get("id") for c in candidates}

    page = int(state.get("last_page", 1))
    index = int(state.get("last_index", 0))

    scanned_batches = 0
    while len(candidates) < MAX_CANDIDATES and scanned_batches < max_scan_batches:
        market_rows, err = fetch_markets_page(page=page, per_page=100)
        if err:
            st.warning(err)
            break
        if not market_rows:
            st.info("더 이상 조회할 코인이 없습니다.")
            break

        for i, coin in enumerate(market_rows[index:], start=index):
            if not apply_base_filters(coin):
                continue
            cid = coin.get("id")
            if cid in existing_ids:
                continue
            candidates.append(build_candidate(coin))
            existing_ids.add(cid)
            if len(candidates) >= MAX_CANDIDATES:
                index = i + 1
                break

        scanned_batches += 1
        if len(candidates) >= MAX_CANDIDATES:
            if index >= len(market_rows):
                page += 1
                index = 0
            break

        page += 1
        index = 0

    save_candidates(candidates)
    save_state({"last_page": page, "last_index": index, "last_updated_utc": utc_now_iso()})
    st.success(f"스캔 완료: 현재 후보 {len(candidates)}개")


def load_market_mode() -> Dict[str, str]:
    btc_prices, btc_err = fetch_daily_market_chart("BTCUSDT", days=400)
    eth_prices, eth_err = fetch_daily_market_chart("ETHUSDT", days=400)

    if btc_err:
        st.warning(btc_err)
    if eth_err:
        st.warning(eth_err)

    btc = compute_ma_status(btc_prices)
    eth = compute_ma_status(eth_prices)
    mode = decide_market_mode(btc["status"], eth["status"])
    return {
        "btc_status": btc["status"],
        "eth_status": eth["status"],
        "mode": mode,
    }


def render_candidates(mode: str) -> None:
    candidates = load_candidates()
    if not candidates:
        st.info("저장된 후보가 없습니다. [Scan next batch] 또는 [Continue]를 눌러주세요.")
        return

    charts: Dict[str, List[List[float]]] = {}
    for c in candidates:
        cid = c.get("id")
        if not cid:
            continue
        prices, err = fetch_daily_market_chart(cid, days=400)
        if err:
            st.caption(f"{c.get('name')} 데이터 일부 미수신: 기본 표시로 대체")
            charts[cid] = []
        else:
            charts[cid] = prices

    enriched = enrich_candidates_with_status(candidates, charts)
    ordered = sort_for_mode(enriched, mode)

    st.subheader("후보 목록")
    for item in ordered:
        st.text(format_candidate_line(item))


controls_col1, controls_col2, controls_col3 = st.columns(3)
max_scan_batches = st.number_input("max_scan_batches", min_value=1, max_value=50, value=5, step=1)

with controls_col1:
    if st.button("Scan next batch", use_container_width=True):
        run_scan(max_scan_batches=1)

with controls_col2:
    if st.button("Continue", use_container_width=True):
        run_scan(max_scan_batches=int(max_scan_batches))

with controls_col3:
    if st.button("Reset", use_container_width=True):
        reset_state()
        reset_candidates()
        st.success("상태와 후보 목록을 초기화했습니다.")

state = load_state()
market = load_market_mode()

panel1, panel2, panel3 = st.columns(3)
panel1.metric("BTC 상태", market["btc_status"])
panel2.metric("ETH 상태", market["eth_status"])
panel3.metric("최종 모드", market["mode"])

st.caption(
    f"마지막 업데이트(UTC): {state.get('last_updated_utc') or '-'} | 데이터 소스: Binance Public API"
)

render_candidates(mode=market["mode"])
