from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
import asyncio
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from core import database
from main import app
from services import ibkr, market, backtests
from api.research import BacktestInput
from test_backtesting import request, wait_run


def data_bar(stamp, close=100):
    return SimpleNamespace(
        date=stamp, open=100, high=101, low=99, close=close, volume=1000
    )


def test_intraday_normalization_preserves_seconds_and_excludes_incomplete():
    now = datetime(2026, 9, 4, 15, 0, 2, tzinfo=timezone.utc)
    bars = [
        data_bar(now),
        data_bar(now - timedelta(seconds=1)),
        data_bar(now - timedelta(seconds=2)),
    ]
    normalized = ibkr.normalize_intraday(bars, 1, now)
    assert [row["time"] for row in normalized] == [
        "2026-09-04T15:00:00Z",
        "2026-09-04T15:00:01Z",
    ]
    assert len(ibkr.normalize_intraday(bars, 5, now)) == 0
    bars[0].volume = float("nan")
    with pytest.raises(HTTPException):
        ibkr.normalize_intraday([bars[0]], 1, now + timedelta(seconds=2))


def test_ticks_preserve_distinct_trades_with_same_timestamp():
    tick = SimpleNamespace(
        time=datetime(2026, 9, 4, 15, tzinfo=timezone.utc),
        price=100,
        size=1,
        exchange="NYSE",
        specialConditions="",
        tickAttribLast=SimpleNamespace(pastLimit=False, unreported=False),
    )
    assert len(ibkr.normalize_ticks([tick, tick])) == 2


def test_pacing_budget_prevents_provider_request():
    import time

    p = ibkr.IBKRProvider()
    p._history_times.extend([time.monotonic()] * 55)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(p._pace_history(("intraday", "AAPL"), True))
    assert exc.value.status_code == 429
    assert int(exc.value.headers["Retry-After"]) > 0


def test_intraday_uses_supported_ibkr_request_and_cache(monkeypatch):
    from test_ibkr import FakeIB

    monkeypatch.setattr(ibkr, "IB", FakeIB)
    provider = ibkr.IBKRProvider()
    try:
        provider._start()
        provider._ib.reqHistoricalDataAsync = AsyncMock(
            return_value=[data_bar(datetime(2026, 9, 4, 15, tzinfo=timezone.utc))]
        )
        result = provider.request("intraday", ("AAPL", "1s", None))
        provider.request("intraday", ("AAPL", "1s", None))
        assert len(result) == 1
        kwargs = provider._ib.reqHistoricalDataAsync.call_args.kwargs
        assert kwargs["barSizeSetting"] == "1 secs"
        assert kwargs["durationStr"] == "1800 S"
        assert kwargs["formatDate"] == 2
        assert provider._ib.reqHistoricalDataAsync.call_count == 1
    finally:
        provider.close()


def test_intraday_backtest_pages_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", str(tmp_path / "intraday.sqlite3"))
    monkeypatch.setattr(market, "DATA_MODE", "demo")
    with TestClient(app) as client:
        snapshot = client.get("/api/stocks/AAPL?interval=1s").json()
        assert snapshot["interval"] == "1s"
        first = snapshot["bars"][0]["time"]
        older = client.get(
            "/api/stocks/AAPL", params={"interval": "1s", "before": first}
        ).json()
        assert older["bars"][-1]["time"] < first
        query = {
            **request("cpp"),
            "interval": "1s",
            "start_date": older["bars"][0]["time"],
            "end_date": snapshot["bars"][-1]["time"],
            "warmup": 5,
        }
        submitted = client.post("/api/backtests", json=query)
        assert submitted.status_code == 202, submitted.json()
        run = wait_run(client, submitted.json()["id"])
        assert run["status"] == "completed", run
        assert run["result"]["data"]["bar_count"] == 999
        assert run["result"]["interval"] == "1s"
        assert run["result"]["metrics"]["sharpe_ratio"] is None
        assert run["result"]["metrics"]["annualized_return_percent"] is None
        assert client.post("/api/backtests", json=query).json()["cache_hit"]
        assert client.get("/api/stocks/AAPL?interval=invalid").status_code == 422
        assert client.get("/api/stocks/AAPL?interval=1s&before=bad").status_code == 422


def test_utc_offset_normalization_and_subsecond_validation():
    q = BacktestInput(
        **{
            **request(),
            "interval": "1s",
            "start_date": "2026-09-04T11:00:00-04:00",
            "end_date": "2026-09-04T15:01:00Z",
        }
    )
    assert q.start_date == "2026-09-04T15:00:00Z"
    with pytest.raises(ValueError):
        BacktestInput(
            **{
                **request(),
                "interval": "1s",
                "start_date": "2026-09-04T15:00:00.5Z",
                "end_date": "2026-09-04T15:00:00Z",
            }
        )
