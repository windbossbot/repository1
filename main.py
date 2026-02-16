from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prices.cache import PriceCache
from prices.yfinance_provider import YFinanceProvider
from screener import run_screen, save_outputs
from universe import UniverseMode, build_universe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="US stock screener")
    parser.add_argument("--mode", choices=[m.value for m in UniverseMode], default=UniverseMode.ALL.value)
    parser.add_argument("--include-etf", action="store_true", help="Include ETFs for NasdaqTrader universes")
    parser.add_argument(
        "--disable-light-name-filter",
        action="store_true",
        help="Disable lightweight security-name filtering for non-common-stock issues",
    )
    parser.add_argument("--disable-sma240-bull-check", action="store_true", help="Do not require SMA120 > SMA240 in bull condition")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--cache-ttl-hours", type=int, default=24)
    parser.add_argument("--log-file", default="logs/screener.log")
    parser.add_argument("--limit", type=int, default=0, help="Optional ticker limit for quick tests")
    return parser.parse_args()


def setup_logging(log_file: str) -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8")],
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_file)

    mode = UniverseMode(args.mode)
    universe = build_universe(
        mode=mode,
        include_etf=args.include_etf,
        light_name_filter=not args.disable_light_name_filter,
    )
    if args.limit > 0:
        universe = universe[: args.limit]

    cache = PriceCache(ttl_hours=args.cache_ttl_hours)
    provider = YFinanceProvider(cache=cache)

    result = run_screen(
        tickers=universe,
        provider=provider,
        require_sma240_for_bull=not args.disable_sma240_bull_check,
        max_workers=args.max_workers,
    )
    bull_path, bear_path = save_outputs(result)

    print(
        " | ".join(
            [
                f"mode={mode.value}",
                f"total={result.total}",
                f"success={result.success}",
                f"fail={result.failed}",
                f"bull={len(result.bull_df)}",
                f"bear={len(result.bear_df)}",
            ]
        )
    )
    print(f"saved: {bull_path} , {bear_path}")
    return 0


if __name__ == "__main__":
    main()
