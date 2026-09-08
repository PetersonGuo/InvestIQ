"""Daily price history, with explicit synthetic demo data and an optional provider."""

import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from readerwriterlock import rwlock
import requests
from fastapi import HTTPException
from core.config import DATA_MODE
from services.resolutions import validate_interval, utc_time

COMPANIES = {
    "AAPL": ("Apple Inc.", 218),
    "MSFT": ("Microsoft Corporation", 425),
    "NVDA": ("NVIDIA Corporation", 132),
    "GOOGL": ("Alphabet Inc.", 176),
    "AMZN": ("Amazon.com Inc.", 198),
    "META": ("Meta Platforms Inc.", 540),
    "TSLA": ("Tesla Inc.", 245),
    "SPY": ("SPDR S&P 500 ETF", 562),
}
_cache = {}

marker = rwlock.RWLockFairD()
read_lock = marker.gen_rlock()
write_lock = marker.gen_wlock()


def provider_get(path, params):
    key = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
    if not key:
        raise HTTPException(503, "Set MASSIVE_API_KEY to use provider data.")
    cache_key = (path, tuple(sorted(params.items())))
    with read_lock:
        cached = _cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 60:
            return cached[1]
    try:
        response = requests.get(
            "https://api.massive.com" + path,
            params=params,
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("status") in {"ERROR", "NOT_AUTHORIZED"}:
            raise ValueError("Provider rejected request")
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(
            502,
            "Market data provider unavailable. Check your API key, plan, and rate limit.",
        ) from exc
    with write_lock:
        if len(_cache) > 256:
            _cache.clear()
        _cache[cache_key] = (time.monotonic(), data)
    return data


def search(ticker="", limit=20):
    if DATA_MODE == "demo":
        return [
            {"ticker": s, "name": n}
            for s, (n, _) in COMPANIES.items()
            if ticker.upper() in s or ticker.lower() in n.lower()
        ][:limit]
    if DATA_MODE == "ibkr":
        from services.ibkr import provider

        return provider.request("search", ticker.strip() or "A")[:limit]
    data = provider_get(
        "/v3/reference/tickers",
        {
            "search": ticker,
            "market": "stocks",
            "active": "true",
            "limit": limit,
            "sort": "ticker",
        },
    )
    return [
        {"ticker": row["ticker"], "name": row.get("name", row["ticker"])}
        for row in data.get("results", [])
    ]


def history(symbol, days=90, before=None, interval="1d"):
    validate_interval(interval)
    if interval != "1d":
        return intraday_history(symbol, interval, before)
    today = date.fromisoformat(before) if before else date.today()
    if DATA_MODE == "demo":
        if symbol not in COMPANIES:
            raise HTTPException(404, "Ticker not available in the demo dataset.")
        seed = sum(map(ord, symbol))
        bars = []
        for offset in range(365, 0, -1):
            day = today - timedelta(days=offset)
            if day.weekday() > 4:
                continue
            t = day.toordinal()
            base = COMPANIES[symbol][1]
            close = round(
                base
                * (1 + 0.06 * math.sin(t / 19 + seed) + 0.018 * math.sin(t / 3 + seed)),
                2,
            )
            opening = round(close * (1 + 0.008 * math.sin(t + seed)), 2)
            bars.append(
                {
                    "time": day.isoformat(),
                    "open": opening,
                    "high": round(max(opening, close) * 1.012, 2),
                    "low": round(min(opening, close) * 0.988, 2),
                    "close": close,
                    "volume": int(15000000 + 9000000 * (1 + math.sin(t + seed))),
                }
            )
        bars = bars[-days:]
    elif DATA_MODE == "ibkr":
        from services.ibkr import provider

        bars = provider.request(
            "history_page" if before else "history",
            (symbol, before) if before else symbol,
        )[-days:]
    else:
        start = today - timedelta(days=days * 2 + 10)
        data = provider_get(
            f"/v2/aggs/ticker/{symbol}/range/1/day/{start}/{today - timedelta(days=1)}",
            {"adjusted": "true", "sort": "asc", "limit": 50000},
        )
        bars = [
            {
                "time": datetime.fromtimestamp(b["t"] / 1000, timezone.utc)
                .date()
                .isoformat(),
                "open": b["o"],
                "high": b["h"],
                "low": b["l"],
                "close": b["c"],
                "volume": b["v"],
            }
            for b in data.get("results", [])
        ][-days:]
    if not bars and before:
        return {"ticker": symbol, "source": DATA_MODE, "bars": [], "has_more": False}
    if not bars:
        raise HTTPException(404, "No price history available for this ticker.")
    last = bars[-1]
    previous = bars[-2]["close"] if len(bars) > 1 else last["open"]
    return {
        "ticker": symbol,
        "name": COMPANIES.get(symbol, (symbol, 0))[0],
        "source": DATA_MODE,
        "as_of": last["time"],
        "price": last["close"],
        "change_percent": round((last["close"] / previous - 1) * 100, 2),
        "bars": bars,
        "has_more": len(bars) >= days,
    }


def intraday_history(symbol, interval, before=None):
    _, _, seconds = validate_interval(interval)
    if DATA_MODE == "ibkr":
        from services.ibkr import provider

        bars = provider.request("intraday", (symbol, interval, before))
    elif DATA_MODE == "demo":
        if symbol not in COMPANIES:
            raise HTTPException(404, "Ticker not available in the demo dataset.")
        # Synthetic regular-session bars, explicitly labeled as demo.
        from zoneinfo import ZoneInfo

        end = utc_time(before) if before else datetime.now(timezone.utc)
        stamp = int(end.timestamp()) // seconds * seconds - seconds
        bars = []
        while len(bars) < 500:
            local = datetime.fromtimestamp(stamp, timezone.utc).astimezone(
                ZoneInfo("America/New_York")
            )
            if local.weekday() < 5 and 570 <= local.hour * 60 + local.minute < 960:
                opening = round(
                    COMPANIES[symbol][1] * (1 + 0.004 * math.sin(stamp / 600)), 4
                )
                close = round(opening * (1 + 0.0002 * math.sin(stamp)), 4)
                bars.append(
                    dict(
                        time=datetime.fromtimestamp(stamp, timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        open=opening,
                        high=max(opening, close) + 0.01,
                        low=min(opening, close) - 0.01,
                        close=close,
                        volume=1000,
                    )
                )
            stamp -= seconds
        bars.reverse()
    else:
        raise HTTPException(422, "Intraday data currently requires IBKR mode.")
    if not bars:
        if not before:
            raise HTTPException(
                404,
                "No intraday bars returned. Check IBKR permissions or choose an earlier end time.",
            )
        return dict(
            ticker=symbol, source=DATA_MODE, bars=[], has_more=False, interval=interval
        )
    last = bars[-1]
    previous = bars[-2]["close"] if len(bars) > 1 else last["open"]
    return dict(
        ticker=symbol,
        name=COMPANIES.get(symbol, (symbol, 0))[0],
        source=DATA_MODE,
        as_of=last["time"],
        price=last["close"],
        change_percent=round((last["close"] / previous - 1) * 100, 2),
        bars=bars,
        has_more=True,
        interval=interval,
    )


def trade_ticks(symbol, before=None):
    if DATA_MODE != "ibkr":
        raise HTTPException(422, "Individual trade records require IBKR mode.")
    from services.ibkr import provider

    ticks = provider.request("ticks", (symbol, "1s", before))
    # Keep every trade in a second; timestamp duplicates are distinct trades.
    return dict(
        ticker=symbol,
        source="ibkr",
        ticks=ticks,
        timestamp_resolution="1 second",
        session="regular",
        as_of=ticks[-1]["time"] if ticks else None,
    )
