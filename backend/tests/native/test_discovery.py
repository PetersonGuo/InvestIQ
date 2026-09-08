import math
import statistics


def test_discovery_and_directions(client):
    response = client.post(
        "/api/pairs/discover",
        json={
            "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "min_correlation": 0,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["pairs"]
    correlations = [p["correlation"] for p in data["pairs"]]
    assert correlations == sorted(correlations, reverse=True)
    for pair in data["pairs"]:
        config = pair["config"]
        assert 0 < config["hedge_ratio"] <= 100
        histories = [
            {
                bar["time"]: bar["close"]
                for bar in client.get(f"/api/stocks/{config[key]}?days=250").json()[
                    "bars"
                ]
            }
            for key in ("ticker_a", "ticker_b")
        ]
        dates = sorted(histories[0].keys() & histories[1].keys())[-61:]
        returns = [
            [math.log(h[b] / h[a]) for a, b in zip(dates, dates[1:])] for h in histories
        ]
        assert math.isclose(
            pair["correlation"], statistics.correlation(*returns), abs_tol=1e-10
        )
        config["threshold"] = 0
        preview = client.post("/api/pairs/preview", json=config).json()
        signal = preview["signal"]
        assert (
            signal["short"]
            == config["ticker_a" if preview["value"] > 0 else "ticker_b"]
        )
        assert (
            signal["long"] == config["ticker_b" if preview["value"] > 0 else "ticker_a"]
        )
        config["condition"] = "inside"
        assert client.post("/api/pairs/preview", json=config).json()["signal"] is None


def test_discovery_validation(client):
    for body in (
        {"tickers": ["AAPL"]},
        {"tickers": ["AAPL", "AAPL"]},
        {"tickers": ["AAPL", "MSFT"], "min_correlation": 1.1},
    ):
        assert client.post("/api/pairs/discover", json=body).status_code == 422


def test_cointegration_and_inverse_pair_signal(client):
    config = {
        "ticker_a": "AAPL",
        "ticker_b": "MSFT",
        "metric": "zscore",
        "condition": "outside",
        "threshold": 0,
        "lookback": 60,
        "hedge_ratio": -1,
    }
    response = client.post("/api/pairs/preview", json=config)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["cointegration"]["observations"] == 60
    assert data["cointegration"]["baseline_end"] == data["baseline_end"]
    assert data["signal"]["legs"][0]["side"] == data["signal"]["legs"][1]["side"]
    assert "not market-neutral" in data["signal"]["reason"]
    scan = client.post(
        "/api/pairs/discover",
        json={
            "tickers": ["AAPL", "MSFT", "NVDA"],
            "min_correlation": 0,
            "relationship": "either",
            "require_cointegration": True,
        },
    )
    assert scan.status_code == 200
    assert all(
        pair["snapshot"]["cointegration"]["reject_no_cointegration_5_percent"]
        for pair in scan.json()["pairs"]
    )
