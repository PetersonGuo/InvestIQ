import concurrent.futures
import json
import sqlite3
import time
from datetime import date, timedelta
import pytest
from conftest import wait_run


def test_native_health_and_history(client):
    assert client.get("/health").json()["backend"] == "cpp"
    assert (
        client.get("/api/search?ticker=Apple").json()["results"][0]["ticker"] == "AAPL"
    )
    first = client.get("/api/stocks/AAPL?days=90").json()
    older = client.get(
        "/api/stocks/AAPL", params={"days": 250, "before": first["bars"][0]["time"]}
    ).json()
    assert len(first["bars"]) == 90 and len(older["bars"]) == 250
    assert older["bars"][-1]["time"] < first["bars"][0]["time"]
    assert client.get("/api/stocks/AAPL?days=0").status_code == 422
    assert client.get("/api/stocks/AAPL?interval=bad").status_code == 422
    assert client.get("/api/stocks/AAPL?before=2025-02-30").status_code == 422
    assert client.get("/api/stocks/UNKNOWN").status_code == 404


@pytest.mark.parametrize("quantity", [0, -1, 1.5, True, "1", None])
def test_invalid_orders(client, quantity):
    assert (
        client.post(
            "/api/order", json={"ticker": "AAPL", "side": "buy", "quantity": quantity}
        ).status_code
        == 422
    )


def test_required_numbers_and_local_security(client):
    assert (
        client.post("/api/order", json={"ticker": "AAPL", "side": "buy"}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/alerts", json={"ticker": "AAPL", "direction": "above"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/order", json={}, headers={"Origin": "https://unrelated.example"}
        ).status_code
        == 403
    )
    assert client.post("/api/order", data="{}").status_code == 415
    assert (
        client.post(
            "/api/order",
            data="{}",
            headers={"Content-Type": "application/json-unrelated"},
        ).status_code
        == 415
    )
    assert (
        client.get("/health", headers={"Host": "unrelated.example"}).status_code == 400
    )


def test_paper_account_transaction_and_persistence(launch):
    client, process, db = launch()
    order = client.post(
        "/api/order", json={"ticker": "AAPL", "side": "buy", "quantity": 3}
    )
    assert order.status_code == 201
    portfolio = client.get("/api/portfolio").json()
    assert (
        portfolio["cash_cents"]
        == 10000000 - order.json()["price_cents"] * 3 - order.json()["fee_cents"]
    )
    assert (
        client.post(
            "/api/order", json={"ticker": "AAPL", "side": "sell", "quantity": 4}
        ).status_code
        == 409
    )
    process.terminate()
    process.wait(timeout=20)
    client, _, _ = launch(database=db)
    assert client.get("/api/portfolio").json()["positions"][0]["quantity"] == 3
    assert (
        client.post(
            "/api/order", json={"ticker": "AAPL", "side": "sell", "quantity": 3}
        ).status_code
        == 201
    )
    assert client.get("/api/portfolio").json()["cash_cents"] < 10000000


def test_concurrent_orders_cannot_overspend(launch):
    client, _, db = launch()
    quote = client.get("/api/stocks/AAPL/quote").json()
    with sqlite3.connect(db) as connection:
        connection.execute(
            "UPDATE account SET cash_cents=?", (round(quote["ask"] * 100) * 100 + 100,)
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(
            pool.map(
                lambda _: client.post(
                    "/api/order",
                    json={"ticker": "AAPL", "side": "buy", "quantity": 300},
                ).status_code,
                range(2),
            )
        )
    assert sorted(statuses) == [201, 409]
    assert client.get("/api/portfolio").json()["cash_cents"] >= 0


def test_price_alert_persistence_and_trigger_on_startup(launch):
    client, process, db = launch()
    response = client.post(
        "/api/alerts", json={"ticker": "AAPL", "direction": "above", "threshold": 1}
    )
    assert response.status_code == 201
    alert = response.json()
    process.terminate()
    process.wait(timeout=20)
    client, _, _ = launch(database=db)
    for _ in range(100):
        row = client.get("/api/alerts").json()[0]
        if not row["active"]:
            break
        time.sleep(0.02)
    assert row["active"] == 0 and row["triggered_at"]
    assert (
        client.put(
            "/api/alerts/" + alert["id"],
            json={"ticker": "AAPL", "direction": "above", "threshold": 10000},
        ).json()["active"]
        == 1
    )
    assert client.delete("/api/alerts/" + alert["id"]).status_code == 204
    assert client.delete("/api/alerts/" + alert["id"]).status_code == 404


def pair_config(**updates):
    return {
        "ticker_a": "AAPL",
        "ticker_b": "MSFT",
        "metric": "ratio",
        "condition": "above",
        "threshold": 0.01,
        "lookback": 60,
        "hedge_ratio": 1,
        "repeat": False,
        **updates,
    }


def test_pair_alerts_versions_events_and_rearming(launch):
    client, process, db = launch()
    assert (
        client.post("/api/pairs/preview", json=pair_config(ticker_b="AAPL")).status_code
        == 422
    )
    assert (
        client.post(
            "/api/pairs/preview", json=pair_config(condition="outside")
        ).status_code
        == 422
    )
    preview = client.post("/api/pairs/preview", json=pair_config()).json()
    assert preview["matched"] and preview["points"]
    alert = client.post("/api/pairs/alerts", json=pair_config()).json()
    process.terminate()
    process.wait(timeout=20)
    client, process, _ = launch(database=db)
    for _ in range(100):
        events = client.get("/api/pairs/events").json()
        if events:
            break
        time.sleep(0.02)
    assert len(events) == 1
    assert client.get("/api/pairs/alerts").json()[0]["active"] == 0
    client.post("/api/pairs/alerts/" + alert["id"] + "/state", json={"active": True})
    process.terminate()
    process.wait(timeout=20)
    client, _, _ = launch(database=db)
    for _ in range(100):
        events = client.get("/api/pairs/events").json()
        if len(events) == 2:
            break
        time.sleep(0.02)
    assert len(events) == 2
    assert client.delete("/api/pairs/alerts/" + alert["id"]).status_code == 204
    assert len(client.get("/api/pairs/events").json()) == 2


def query(client, language="cpp", preset="moving_average"):
    p = next(
        p
        for p in client.get("/api/strategies/examples").json()
        if p["language"] == language and p["template_id"] == preset
    )
    return {
        **p,
        "ticker": "AAPL",
        "start_date": (date.today() - timedelta(days=365)).isoformat(),
        "end_date": (date.today() - timedelta(days=1)).isoformat(),
        "initial_cash": 100000,
        "commission": 1,
        "slippage_bps": 5,
    }


@pytest.mark.parametrize(
    "preset", ["moving_average", "buy_and_hold", "channel_breakout", "mean_reversion"]
)
def test_native_jobs_python_cpp_parity(client, preset):
    outputs = []
    for lang in ("python", "cpp"):
        q = query(client, lang, preset)
        saved = client.post("/api/strategies", json=q)
        assert saved.status_code == 201
        submitted = client.post("/api/backtests", json=q)
        assert submitted.status_code == 202, submitted.text
        run = wait_run(client, submitted.json()["id"])
        assert run["status"] == "completed", run
        assert run["result"]["engine_version"] == "3.0.0-native-cpp"
        outputs.append(run["result"])
    assert outputs[0]["fills"] == outputs[1]["fills"]
    assert outputs[0]["equity_curve"] == outputs[1]["equity_curve"]
    assert outputs[0]["metrics"] == outputs[1]["metrics"]


def test_cache_restart_bypass_changed_inputs_and_failed_runs(launch):
    client, process, db = launch()
    q = query(client)
    submitted = client.post("/api/backtests", json=q).json()
    assert wait_run(client, submitted["id"])["status"] == "completed"
    process.terminate()
    process.wait(timeout=20)
    client, _, _ = launch(database=db)
    cached = client.post("/api/backtests", json={**q, "name": "Renamed"}).json()
    assert cached["cache_hit"] and cached["id"] == submitted["id"]
    fresh = client.post("/api/backtests", json={**q, "force_rerun": True}).json()
    assert fresh["id"] != cached["id"] and not fresh["cache_hit"]
    assert wait_run(client, fresh["id"])["status"] == "completed"
    changed = client.post("/api/backtests", json={**q, "commission": 2}).json()
    assert not changed["cache_hit"]
    assert wait_run(client, changed["id"])["status"] == "completed"
    failed = []
    for _ in range(2):
        bad = client.post(
            "/api/backtests", json={**q, "code": "invalid c++ !!!"}
        ).json()
        assert wait_run(client, bad["id"])["status"] == "failed"
        failed.append(bad["id"])
    assert failed[0] != failed[1]


def test_intraday_jobs_page_exact_range(client):
    snapshot = client.get("/api/stocks/AAPL?interval=1s").json()
    older = client.get(
        "/api/stocks/AAPL",
        params={"interval": "1s", "before": snapshot["bars"][0]["time"]},
    ).json()
    assert older["bars"][-1]["time"] < snapshot["bars"][0]["time"]
    q = {
        **query(client),
        "interval": "1s",
        "start_date": older["bars"][0]["time"],
        "end_date": snapshot["bars"][-1]["time"],
    }
    submitted = client.post("/api/backtests", json=q).json()
    run = wait_run(client, submitted["id"])
    assert run["status"] == "completed", run
    assert run["result"]["data"]["bar_count"] == 999
    assert run["result"]["metrics"]["sharpe_ratio"] is None
    assert run["result"]["metrics"]["annualized_return_percent"] is None


def test_unavailable_broker_does_not_fall_back_to_demo(launch):
    client, _, _ = launch("ibkr", IBKR_PORT="1", IBKR_CLIENT_ID="99")
    r = client.get("/api/stocks/AAPL")
    assert r.status_code in (502, 503, 504)
    assert "IBKR" in r.json()["detail"]
