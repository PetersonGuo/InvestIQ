import pytest
from conftest import wait_run
from test_http import query


@pytest.mark.parametrize("preset", ["equal_weight", "pairs_mean_reversion"])
def test_portfolio_languages_and_benchmarks(client, preset):
    results = []
    for language in ("cpp", "python"):
        q = query(client, language, preset)
        response = client.post("/api/backtests", json=q)
        assert response.status_code == 202, response.text
        run = wait_run(client, response.json()["id"])
        assert run["status"] == "completed", run
        result = run["result"]
        results.append(result)
        assert result["data"]["benchmark_ticker"] == "SPY"
        assert len(result["data"]["assets"]) >= 2
        assert all("spy" in p for p in result["equity_curve"])
        assert result["risk_comparison"]["spy"]["beta_to_spy"] == pytest.approx(1)
        assert result["risk_comparison"]["strategy"]["max_drawdown_percent"] >= 0
    assert results[0]["metrics"] == results[1]["metrics"]
    assert results[0]["fills"] == results[1]["fills"]


def test_single_symbol_spy_same_period(client):
    q = query(client, preset="buy_and_hold")
    q["ticker"] = "SPY"
    response = client.post("/api/backtests", json=q)
    run = wait_run(client, response.json()["id"])
    assert run["status"] == "completed", run
    assert all(p["equity"] == p["spy"] for p in run["result"]["equity_curve"])


def test_invalid_portfolio_symbols(client):
    q = query(client)
    for tickers in ([], ["AAPL", "AAPL"], ["AAPL", True], ["AAPL"] * 11):
        assert (
            client.post("/api/backtests", json={**q, "tickers": tickers}).status_code
            == 422
        )


def test_portfolio_cache_includes_every_symbol(client):
    q = query(client, preset="equal_weight")
    q["tickers"] = ["AAPL", "MSFT"]
    first = client.post("/api/backtests", json=q).json()
    done = wait_run(client, first["id"])
    assert done["status"] == "completed"
    replay = client.post("/api/backtests", json=q).json()
    assert replay["cache_hit"] is True and replay["id"] == first["id"]
    changed = client.post(
        "/api/backtests", json={**q, "tickers": ["AAPL", "NVDA"]}
    ).json()
    assert changed["id"] != first["id"]
    result = wait_run(client, changed["id"])
    assert result["status"] == "completed"
    assert result["result"]["data"]["tickers"] == ["AAPL", "NVDA"]


def test_portfolio_rejects_excessive_exposure(client):
    q = query(client, "python", "equal_weight")
    q["code"] = (
        "class Strategy:\n    def on_bar(self, ctx):\n        return {symbol: 1 for symbol in ctx.histories}"
    )
    response = client.post("/api/backtests", json=q).json()
    result = wait_run(client, response["id"])
    assert result["status"] == "failed" and "gross exposure" in result["error"]


def test_saved_strategy_restores_universe(client):
    q = query(client, preset="equal_weight")
    saved = client.post("/api/strategies", json=q)
    assert saved.status_code == 201, saved.text
    strategies = client.get("/api/strategies").json()
    row = next(row for row in strategies if row["id"] == saved.json()["id"])
    assert row["tickers"] == q["tickers"]
