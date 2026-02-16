from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from indicators import compute_sma_metrics, is_bear, is_bull
from prices.provider import PriceProvider
from universe import TickerRecord


@dataclass
class ScreenResult:
    bull_df: pd.DataFrame
    bear_df: pd.DataFrame
    total: int
    success: int
    failed: int


def run_screen(
    tickers: list[TickerRecord],
    provider: PriceProvider,
    require_sma240_for_bull: bool = True,
    max_workers: int = 8,
    throttle_seconds: float = 0.15,
) -> ScreenResult:
    logger = logging.getLogger(__name__)
    bull_rows: list[dict] = []
    bear_rows: list[dict] = []
    success = 0
    failed = 0

    def _worker(rec: TickerRecord) -> tuple[TickerRecord, dict[str, float] | None, str | None]:
        try:
            bars = provider.get_daily_bars(rec.ticker)
            metrics = compute_sma_metrics(bars)
            if metrics is None:
                return rec, None, "not enough data"
            return rec, metrics, None
        except Exception as exc:
            return rec, None, str(exc)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for rec in tickers:
            futures[pool.submit(_worker, rec)] = rec
            time.sleep(throttle_seconds)

        for future in as_completed(futures):
            rec, metrics, err = future.result()
            if err is not None:
                failed += 1
                logger.warning("ticker failed: %s (%s)", rec.ticker, err)
                continue

            success += 1
            row = {
                "ticker": rec.ticker,
                "name": rec.name,
                "exchange": rec.exchange,
                **metrics,
            }
            if is_bull(metrics, require_sma240=require_sma240_for_bull):
                bull_rows.append(row)
            if is_bear(metrics):
                bear_rows.append(row)

    bull_df = pd.DataFrame(bull_rows).sort_values("distance_to_sma20_pct", ascending=True) if bull_rows else pd.DataFrame()
    bear_df = pd.DataFrame(bear_rows).sort_values("distance_to_sma120_pct", ascending=True) if bear_rows else pd.DataFrame()

    return ScreenResult(
        bull_df=bull_df,
        bear_df=bear_df,
        total=len(tickers),
        success=success,
        failed=failed,
    )


def save_outputs(result: ScreenResult, output_dir: str | Path = "output") -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bull_path = out_dir / "bull.csv"
    bear_path = out_dir / "bear.csv"

    result.bull_df.to_csv(bull_path, index=False)
    result.bear_df.to_csv(bear_path, index=False)
    return bull_path, bear_path
