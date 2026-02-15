from __future__ import annotations


def get_crypto_bases(exchange_id: str = "binance") -> list[dict]:
    import ccxt

    ex_cls = getattr(ccxt, exchange_id)
    ex = ex_cls({"enableRateLimit": True})
    markets = ex.load_markets()
    allowed_quotes = {"USDT", "BTC", "ETH"}
    seen = set()
    items = []
    for m in markets.values():
        base = m.get("base")
        quote = m.get("quote")
        if not base or not quote or quote not in allowed_quotes:
            continue
        if base in seen:
            continue
        seen.add(base)
        items.append({"symbol": base, "name": base})
    return items
