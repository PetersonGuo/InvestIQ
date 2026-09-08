import concurrent.futures
import math


def test_live_quote_and_spread_fees(client):
    quote = client.get("/api/stocks/AAPL/quote").json()
    assert quote["source"] == "demo"
    assert quote["bid"] < quote["ask"]
    buy = client.post(
        "/api/order", json={"ticker": "AAPL", "side": "buy", "quantity": 2}
    ).json()
    assert buy["price_cents"] >= quote["ask"] * 100 - 1e-8
    sell = client.post(
        "/api/order", json={"ticker": "AAPL", "side": "sell", "quantity": 2}
    ).json()
    assert sell["price_cents"] <= quote["bid"] * 100 + 1e-8
    assert (
        client.get("/api/portfolio").json()["cash_cents"]
        == 10000000
        + 2 * (sell["price_cents"] - buy["price_cents"])
        - buy["fee_cents"]
        - sell["fee_cents"]
    )


def test_ioc_limits_and_partial_fills(client):
    rejected = client.post(
        "/api/order",
        json={
            "ticker": "AAPL",
            "side": "buy",
            "quantity": 2,
            "order_type": "limit",
            "limit_price": 0.01,
        },
    )
    assert rejected.status_code == 409
    assert not client.get("/api/portfolio").json()["orders"]
    quote = client.get("/api/stocks/AAPL/quote").json()
    fill = client.post(
        "/api/order",
        json={
            "ticker": "AAPL",
            "side": "buy",
            "quantity": 200,
            "order_type": "limit",
            "limit_price": math.ceil(quote["ask"] * 100) / 100,
        },
    ).json()
    assert fill["quantity"] == 100
    assert fill["cancelled_quantity"] == 100
    assert fill["status"] == "partially_filled"
    assert fill["time_in_force"] == "IOC"


def test_idempotent_order_concurrent_and_restart(launch):
    client, process, db = launch()
    body = {"ticker": "AAPL", "side": "buy", "quantity": 1, "request_id": "one-order"}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fills = list(
            pool.map(lambda _: client.post("/api/order", json=body).json(), range(2))
        )
    assert fills[0]["id"] == fills[1]["id"]
    assert client.get("/api/portfolio").json()["positions"][0]["quantity"] == 1
    process.terminate()
    process.wait(timeout=20)
    client, _, _ = launch(database=db)
    assert client.post("/api/order", json=body).json()["id"] == fills[0]["id"]
    assert client.post("/api/order", json={**body, "quantity": 2}).status_code == 409
    assert len(client.get("/api/portfolio").json()["orders"]) == 1
