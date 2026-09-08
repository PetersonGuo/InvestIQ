import concurrent.futures
import pytest
from fastapi.testclient import TestClient
from core import database
from main import app
from services import market
from services.alerts import check_alerts


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setattr(market, "DATA_MODE", "demo")
    with TestClient(app) as client:
        yield client


def test_search_and_history(client):
    assert client.get("/health").status_code == 200
    assert (
        client.get("/api/search?ticker=apple").json()["results"][0]["ticker"] == "AAPL"
    )
    assert client.post("/api/search", json={"ticker": "a"}).status_code == 200
    stock = client.get("/api/stocks/AAPL?days=20").json()
    assert stock["source"] == "demo"
    assert len(stock["bars"]) == 20
    assert stock["price"] == stock["bars"][-1]["close"]
    assert all(
        b["low"]
        <= min(b["open"], b["close"])
        <= max(b["open"], b["close"])
        <= b["high"]
        for b in stock["bars"]
    )
    assert client.get("/api/stocks/UNKNOWN").status_code == 404
    assert client.get("/api/stocks/AAPL?days=0").status_code == 422


def test_buy_sell_and_balance(client):
    order = client.post(
        "/api/order", json={"ticker": "AAPL", "side": "buy", "quantity": 3}
    )
    assert order.status_code == 201
    price = order.json()["price_cents"]
    portfolio = client.get("/api/portfolio").json()
    assert portfolio["cash_cents"] == 10000000 - price * 3
    assert portfolio["positions"][0]["quantity"] == 3
    assert (
        client.post(
            "/api/order", json={"ticker": "AAPL", "side": "sell", "quantity": 4}
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/api/order", json={"ticker": "AAPL", "side": "buy", "quantity": 1000000}
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/api/order", json={"ticker": "AAPL", "side": "sell", "quantity": 3}
        ).status_code
        == 201
    )
    portfolio = client.get("/api/portfolio").json()
    assert portfolio["cash_cents"] == 10000000
    assert portfolio["positions"] == []
    assert len(portfolio["orders"]) == 2


@pytest.mark.parametrize("quantity", [0, -1, 1.5, True, "1"])
def test_bad_orders(client, quantity):
    assert (
        client.post(
            "/api/order", json={"ticker": "AAPL", "side": "buy", "quantity": quantity}
        ).status_code
        == 422
    )


def test_alert_lifecycle(client):
    payload = {"ticker": "AAPL", "direction": "above", "threshold": 1}
    response = client.post("/api/alerts", json=payload)
    assert response.status_code == 201
    alert_id = response.json()["id"]
    check_alerts()
    row = client.get("/api/alerts").json()[0]
    assert row["active"] == 0 and row["triggered_at"]
    check_alerts()
    assert client.get("/api/alerts").json()[0]["triggered_at"] == row["triggered_at"]
    assert (
        client.put(
            "/api/alerts/" + alert_id, json={**payload, "threshold": 100000}
        ).json()["active"]
        == 1
    )
    check_alerts()
    assert client.get("/api/alerts").json()[0]["active"] == 1
    assert client.delete("/api/alerts/" + alert_id).status_code == 204
    assert client.delete("/api/alerts/" + alert_id).status_code == 404
    assert client.get("/api/alerts").json() == []


def test_bad_alert(client):
    assert (
        client.post(
            "/api/alerts",
            json={"ticker": "AAPL", "direction": "above", "threshold": -1},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/alerts",
            json={"ticker": "UNKNOWN", "direction": "above", "threshold": 1},
        ).status_code
        == 404
    )


def test_persistence(client):
    client.post("/api/order", json={"ticker": "MSFT", "side": "buy", "quantity": 1})
    database.initialize()
    assert client.get("/api/portfolio").json()["positions"][0]["ticker"] == "MSFT"


def test_concurrent_orders_cannot_overspend(client):
    def buy(_):
        return client.post(
            "/api/order", json={"ticker": "AAPL", "side": "buy", "quantity": 300}
        ).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(buy, range(2)))
    assert sorted(statuses) == [201, 409]
    assert client.get("/api/portfolio").json()["cash_cents"] >= 0


def test_provider_failure_is_explicit(client, monkeypatch):
    monkeypatch.setattr(market, "DATA_MODE", "massive")
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    assert client.get("/api/stocks/AAPL").status_code == 503
