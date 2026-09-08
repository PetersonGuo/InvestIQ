from datetime import timedelta
from math import exp
from statistics import mean, pstdev
from zoneinfo import ZoneInfo
import concurrent.futures

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from api.pairs import PairConfig
from core import database
from main import app
from services import market, pairs


def config(**kwargs):
    return PairConfig(
        ticker_a="AAPL", ticker_b="MSFT", lookback=20, **kwargs
    ).model_dump()


def histories(count=21):
    today = pairs.datetime.now(ZoneInfo("America/New_York")).date()
    dates = [(today - timedelta(days=count - i)).isoformat() for i in range(count)]
    a = [
        {
            "time": day,
            "close": 100 * exp(0.1 if i == count - 1 else 0.01 * (-1 if i % 2 else 1)),
        }
        for i, day in enumerate(dates)
    ]
    b = [{"time": day, "close": 100.0} for day in dates]
    return a, b


def test_zscore_uses_previous_aligned_bars():
    a, b = histories()
    snapshot = pairs.calculate(config(), a, b)
    assert snapshot["z_score"] == pytest.approx(10)
    assert snapshot["baseline_end"] == a[-2]["time"]
    assert snapshot["baseline_start"] == a[0]["time"]
    assert snapshot["matched"] is True
    assert snapshot["ratio"] == pytest.approx(exp(0.1))
    assert snapshot["points"][-1]["value"] == snapshot["value"]


def test_ratio_ignores_weight_and_missing_dates_are_aligned():
    a, b = histories(22)
    del b[4]
    snapshot = pairs.calculate(
        config(metric="ratio", condition="above", threshold=1, hedge_ratio=2), a, b
    )
    assert snapshot["value"] == pytest.approx(exp(0.1))
    assert snapshot["aligned_bars"] == 21
    assert snapshot["z_score"] is None
    assert pairs.calculate(config(), a, b)["aligned_bars"] == 21


@pytest.mark.parametrize("problem", ["mismatch", "stale", "short", "flat", "invalid"])
def test_bad_pair_data_does_not_evaluate(problem):
    a, b = histories()
    if problem == "mismatch":
        b.pop()
    if problem == "stale":
        for rows in (a, b):
            for bar in rows:
                bar["time"] = (
                    pairs.date.fromisoformat(bar["time"]) - timedelta(days=20)
                ).isoformat()
    if problem == "short":
        a = a[-10:]
        b = b[-10:]
    if problem == "flat":
        a = [dict(row) for row in b]
    if problem == "invalid":
        a[-1]["close"] = float("nan")
    with pytest.raises(HTTPException):
        pairs.calculate(config(), a, b)


@pytest.mark.parametrize(
    "condition,value,threshold,expected",
    [
        ("above", 2, 2, True),
        ("below", -2, -2, True),
        ("outside", -2.1, 2, True),
        ("inside", -0.4, 0.5, True),
        ("inside", 0.6, 0.5, False),
        ("outside", 1, 2, False),
    ],
)
def test_conditions(condition, value, threshold, expected):
    assert (
        pairs.matches(value, {"condition": condition, "threshold": threshold})
        is expected
    )


@pytest.fixture
def pair_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", str(tmp_path / "pairs.sqlite3"))
    pairs.initialize()
    a, b = histories()
    monkeypatch.setattr(
        market,
        "history",
        lambda symbol, days: {"bars": a if symbol == "AAPL" else b, "source": "demo"},
    )
    return monkeypatch


def test_one_shot_persistence_rearm_and_notifications(pair_db):
    alert = pairs.save(config())
    pairs.check_pair_alerts()
    assert pairs.get(alert["id"])["active"] == 0
    assert len(pairs.events()) == 1
    pairs.check_pair_alerts()
    pairs.initialize()
    assert len(pairs.events()) == 1
    pairs.set_active(alert["id"], True)
    pairs.check_pair_alerts()
    assert len(pairs.events()) == 2
    pairs.delete(alert["id"])
    assert pairs.list_alerts() == []
    assert len(pairs.events()) == 2


def test_repeat_needs_clear_on_later_bar(pair_db):
    alert = pairs.save(config(repeat=True))
    snapshot = pairs.analyze(config(repeat=True))
    pair_db.setattr(pairs, "analyze", lambda *args: snapshot.copy())
    pairs.check_pair_alerts()
    assert len(pairs.events()) == 1
    assert pairs.get(alert["id"])["active"] == 1
    snapshot["matched"] = False
    pairs.check_pair_alerts()  # revisions of the same close do not rearm
    snapshot["matched"] = True
    snapshot["as_of"] = "2090-01-01"
    pairs.check_pair_alerts()
    assert len(pairs.events()) == 1
    snapshot["matched"] = False
    snapshot["as_of"] = "2090-01-02"
    pairs.check_pair_alerts()
    snapshot["matched"] = True
    snapshot["as_of"] = "2090-01-03"
    pairs.check_pair_alerts()
    assert len(pairs.events()) == 2
    pairs.check_pair_alerts()
    assert len(pairs.events()) == 2


def test_failures_pause_evaluation_and_recover(pair_db):
    alert = pairs.save(config())
    original = pairs.analyze

    def broken(*args):
        raise HTTPException(503, "IBKR offline")

    pair_db.setattr(pairs, "analyze", broken)
    pairs.check_pair_alerts()
    assert pairs.get(alert["id"])["last_error"] == "IBKR offline"
    assert pairs.events() == []
    pair_db.setattr(pairs, "analyze", original)
    pairs.check_pair_alerts()
    assert pairs.get(alert["id"])["last_error"] is None
    assert len(pairs.events()) == 1


def test_concurrent_checks_do_not_duplicate(pair_db):
    pairs.save(config(repeat=True))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: pairs.check_pair_alerts(), range(2)))
    assert len(pairs.events()) == 1


def test_edit_during_evaluation_cannot_trigger_old_rule(pair_db):
    alert = pairs.save(config())
    original = pairs.analyze

    def edited(cfg, cache=None):
        snapshot = original(cfg, cache)
        pairs.set_active(alert["id"], False)
        return snapshot

    pair_db.setattr(pairs, "analyze", edited)
    pairs.check_pair_alerts()
    assert pairs.get(alert["id"])["active"] == 0
    assert pairs.events() == []


def test_pair_api_validation_and_crud(pair_db):
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/pairs/preview", json={**config(), "ticker_b": "AAPL"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/pairs/preview", json={**config(), "metric": "ratio"}
            ).status_code
            == 422
        )
        assert client.post("/api/pairs/preview", json=config()).status_code == 200
        created = client.post("/api/pairs/alerts", json=config())
        assert created.status_code == 201
        alert_id = created.json()["id"]
        updated = client.put(
            "/api/pairs/alerts/" + alert_id, json={**config(), "threshold": 15}
        )
        assert updated.status_code == 200
        assert updated.json()["threshold"] == 15
        assert (
            client.post(
                f"/api/pairs/alerts/{alert_id}/state", json={"active": False}
            ).json()["active"]
            == 0
        )
        assert client.delete("/api/pairs/alerts/" + alert_id).status_code == 204
        assert client.delete("/api/pairs/alerts/" + alert_id).status_code == 404
        assert client.get("/api/pairs/alerts").json() == []
