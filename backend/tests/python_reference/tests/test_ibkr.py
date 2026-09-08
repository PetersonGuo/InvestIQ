import asyncio
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from services import ibkr, market


def bar(day, close=102):
    return SimpleNamespace(
        date=day, open=100, high=105, low=99, close=close, volume=1500
    )


def test_normalization_excludes_incomplete_sessions_and_sorts():
    result = ibkr.normalize_bars(
        [bar("20260904"), bar(date(2026, 9, 3)), bar("20260907")],
        today=date(2026, 9, 7),
    )
    assert [b["time"] for b in result] == ["2026-09-03", "2026-09-04"]
    assert result[0]["close"] == 102


def test_invalid_ibkr_prices_rejected():
    with pytest.raises(HTTPException) as exc:
        ibkr.normalize_bars([bar("20200101", float("nan"))])
    assert exc.value.status_code == 502


class FakeIB:
    def __init__(self):
        self.connected = False
        self.connections = 0
        self.history_requests = 0
        self.client = SimpleNamespace(connectAsync=self.connect)
        self.qualifyContractsAsync = AsyncMock(
            return_value=[SimpleNamespace(symbol="AAPL")]
        )
        self.reqMatchingSymbolsAsync = AsyncMock(
            return_value=[
                SimpleNamespace(
                    contract=SimpleNamespace(
                        symbol="AAPL",
                        secType="STK",
                        currency="USD",
                        description="Apple",
                    )
                )
            ]
        )

    async def connect(self, host, port, client_id, timeout):
        self.connections += 1
        self.connected = True

    def isConnected(self):
        return self.connected

    def disconnect(self):
        self.connected = False

    async def reqHistoricalDataAsync(self, contract, **kwargs):
        self.history_requests += 1
        assert kwargs["barSizeSetting"] == "1 day"
        assert kwargs["useRTH"] is True
        return [bar("20200102"), bar("20200103")]


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(ibkr, "IB", FakeIB)
    p = ibkr.IBKRProvider()
    yield p
    p.close()


def test_transport_cache_search_and_reconnect(provider):
    assert provider.request("history", "AAPL")[0]["time"] == "2020-01-02"
    assert len(provider.request("history", "AAPL")) == 2
    assert provider._ib.history_requests == 1
    assert provider.request("search", "Apple") == [{"ticker": "AAPL", "name": "Apple"}]
    provider._ib.connected = False
    provider.request("history", "AAPL")
    assert provider._ib.connections == 2
    assert provider._ib.history_requests == 2
    assert provider.status()["connected"] is True


def test_connection_refusal_is_actionable(provider):
    provider._start()
    provider._ib.client.connectAsync = AsyncMock(side_effect=ConnectionRefusedError())
    with pytest.raises(HTTPException) as exc:
        provider.request("history", "AAPL")
    assert exc.value.status_code == 503
    assert "Read-Only API" in exc.value.detail
    assert provider.status()["connected"] is False


def test_permissions_error_is_not_demo_fallback(provider):
    provider._start()
    provider._ib.reqHistoricalDataAsync = AsyncMock(
        side_effect=ibkr.RequestError(1, 162, "No market data permissions")
    )
    with pytest.raises(HTTPException) as exc:
        provider.request("history", "AAPL")
    assert exc.value.status_code == 502
    assert "162" in exc.value.detail
    assert "subscriptions" in exc.value.detail


def test_market_routes_use_ibkr(provider, monkeypatch):
    monkeypatch.setattr(market, "DATA_MODE", "ibkr")
    monkeypatch.setattr(ibkr, "provider", provider)
    result = market.history("AAPL", 2)
    assert result["source"] == "ibkr"
    assert result["price"] == 102
    assert market.search("Apple")[0]["ticker"] == "AAPL"


def test_search_timeout_is_not_empty_success(provider):
    provider._start()
    provider._ib.reqMatchingSymbolsAsync = AsyncMock(return_value=None)
    with pytest.raises(HTTPException) as exc:
        provider.request("search", "Apple")
    assert exc.value.status_code == 504
