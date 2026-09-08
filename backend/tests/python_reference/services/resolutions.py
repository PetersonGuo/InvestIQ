"""IBKR bar sizes and bounded historical pages (regular US stock sessions)."""

from datetime import datetime, timezone
from fastapi import HTTPException

# interval: (IBKR bar size, IBKR duration, seconds per bar)
RESOLUTIONS = {
    "1s": ("1 secs", "1800 S", 1),
    "5s": ("5 secs", "3600 S", 5),
    "10s": ("10 secs", "14400 S", 10),
    "15s": ("15 secs", "14400 S", 15),
    "30s": ("30 secs", "1 D", 30),
    "1m": ("1 min", "1 D", 60),
    "2m": ("2 mins", "2 D", 120),
    "3m": ("3 mins", "1 W", 180),
    "5m": ("5 mins", "1 W", 300),
    "15m": ("15 mins", "1 W", 900),
    "30m": ("30 mins", "1 M", 1800),
    "1h": ("1 hour", "1 M", 3600),
    "4h": ("4 hours", "1 M", 14400),
    "1d": ("1 day", "2 Y", 86400),
}


def utc_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def validate_interval(interval):
    if interval not in RESOLUTIONS:
        raise HTTPException(422, "Unsupported candle interval.")
    return RESOLUTIONS[interval]
