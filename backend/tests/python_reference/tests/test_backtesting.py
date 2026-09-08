import time
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient
from core import database
from main import app
from services import backtests, market
from simulation.engine import simulate


def bars(prices=(10, 20, 30, 25, 15, 20, 30, 40)):
    return [
        {
            "time": (date(2025, 1, 1) + timedelta(days=i)).isoformat(),
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": 1000,
        }
        for i, price in enumerate(prices)
    ]


def test_no_future_bars_and_next_open_fills():
    observed = []

    def strategy(ctx):
        observed.append(len(ctx.history))
        return 1 if len(ctx.history) == 1 else None

    result = simulate(
        bars((10, 20, 30)), strategy, initial_cash=100, commission=0, slippage_bps=0
    )
    assert observed == [1, 2, 3]
    assert result["fills"][0]["signal_date"] == "2025-01-01"
    assert result["fills"][0]["time"] == "2025-01-02"
    assert result["fills"][0]["price"] == 20
    assert result["fills"][0]["quantity"] == 5
    assert result["metrics"]["final_equity"] == 150
    assert result["metrics"]["benchmark_return_percent"] == 50


def test_fees_slippage_and_round_trip_accounting():
    result = simulate(
        bars((100, 100, 100)),
        lambda ctx: 1 if len(ctx.history) == 1 else 0,
        initial_cash=1000,
        commission=2,
        slippage_bps=100,
    )
    assert result["fills"][0]["quantity"] == 9
    assert result["fills"][0]["price"] == 101
    assert result["fills"][1]["price"] == 99
    assert result["metrics"]["final_equity"] == 978
    assert result["metrics"]["total_commission"] == 4
    assert result["metrics"]["open_shares"] == 0
    assert result["fills"][1]["realized_pnl"] == -22


def test_warmup_and_final_signal_do_not_fill_early():
    result = simulate(
        bars((10, 10, 10)),
        lambda ctx: 1,
        initial_cash=100,
        commission=0,
        slippage_bps=0,
        warmup=1,
    )
    assert len(result["fills"]) == 1
    assert result["fills"][0]["time"] == "2025-01-03"
    assert result["equity_curve"][0]["shares"] == 0
    assert result["unfilled_final_signal"] == 1


def test_drawdown_and_no_trades():
    result = simulate(
        bars((10, 10, 5)),
        lambda ctx: 1 if len(ctx.history) == 1 else None,
        initial_cash=100,
        commission=0,
        slippage_bps=0,
    )
    assert result["metrics"]["max_drawdown_percent"] == 50
    flat = simulate(bars(), lambda ctx: None)
    assert flat["metrics"]["total_return_percent"] == 0
    assert flat["metrics"]["win_rate_percent"] is None
    assert flat["metrics"]["sharpe_ratio"] is None


@pytest.mark.parametrize("value", [float("nan"), -1, 1.1, True, "buy"])
def test_invalid_target_fails(value):
    with pytest.raises(ValueError, match="target weight"):
        simulate(bars(), lambda ctx: value)


def request(language="python", code=None):
    suffix = "py" if language == "python" else "cpp"
    return {
        "name": "Test",
        "language": language,
        "code": code
        or (
            backtests.ROOT / "simulation/examples" / f"moving_average.{suffix}"
        ).read_text(),
        "params": {"fast": 2, "slow": 3},
        "initial_cash": 1000,
        "commission": 1,
        "slippage_bps": 5,
        "warmup": 0,
        "ticker": "AAPL",
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
    }


def test_python_cpp_parity():
    python_result = backtests.execute_strategy(request(), bars())
    cpp_result = backtests.execute_strategy(request("cpp"), bars())
    assert python_result["fills"] == cpp_result["fills"]
    assert python_result["equity_curve"] == cpp_result["equity_curve"]
    assert python_result["metrics"] == cpp_result["metrics"]


def test_python_error_and_cpp_compile_error():
    with pytest.raises(ValueError, match="SyntaxError"):
        backtests.execute_strategy(request(code="not valid python !!!"), bars())
    with pytest.raises(ValueError, match="exited with code"):
        backtests.execute_strategy(request("cpp", "not valid c++ !!!"), bars())


def test_native_crash_is_contained():
    code = '#include "strategy_api.h"\n#include <cstdlib>\nextern "C" double on_bar(const SA_Bar*, int, double, int64_t, const char*) { std::abort(); }'
    with pytest.raises(ValueError, match="exited with code"):
        backtests.execute_strategy(request("cpp", code), bars())


def test_run_timeout_terminates_process(tmp_path):
    import sys

    with pytest.raises(ValueError, match="time limit"):
        backtests.run_process(
            [sys.executable, "-c", "while True: pass"],
            tmp_path,
            0.2,
            tmp_path / "out.log",
        )


def test_job_persistence_and_strategy_save(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", str(tmp_path / "research.sqlite3"))
    monkeypatch.setattr(
        market, "history", lambda *args: {"source": "demo", "bars": bars()}
    )
    with TestClient(app) as client:
        saved = client.post("/api/strategies", json=request())
        assert saved.status_code == 201
        assert client.get("/api/strategies").json()[0]["code"] == request()["code"]
        response = client.post("/api/backtests", json=request())
        assert response.status_code == 202
        run_id = response.json()["id"]
        for _ in range(100):
            run = client.get("/api/backtests/" + run_id).json()
            if run["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert run["status"] == "completed", run
        assert run["result"]["data"]["source"] == "demo"
        assert run["request"]["code"] == request()["code"]
        assert run["result"]["data"]["sha256"]
        assert client.get("/api/backtests").json()[0]["id"] == run_id
        backtests.initialize()
        assert client.get("/api/backtests/" + run_id).json()["status"] == "completed"
        assert (
            client.post(
                "/api/backtests", json={**request(), "end_date": "2020-01-01"}
            ).status_code
            == 422
        )


def test_code_execution_api_rejects_cross_origin(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", str(tmp_path / "security.sqlite3"))
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/backtests",
                json=request(),
                headers={"Origin": "https://unrelated.example"},
            ).status_code
            == 403
        )
        assert client.post("/api/backtests", content="{}").status_code == 415
        assert (
            client.post(
                "/api/backtests", json=request(), headers={"Host": "unrelated.example"}
            ).status_code
            == 400
        )


def test_extreme_short_window_does_not_overflow_report():
    data = bars((1, 1, 1000000))
    data[0]["low"] = 0.5
    data[1]["low"] = 0.5
    result = simulate(
        data, lambda ctx: 1, initial_cash=100, commission=0, slippage_bps=0
    )
    assert result["metrics"]["annualized_return_percent"] is None
    assert result["metrics"]["final_equity"] == 100000000


@pytest.mark.parametrize(
    "preset_id",
    ["moving_average", "buy_and_hold", "channel_breakout", "mean_reversion"],
)
def test_all_preset_languages_match(preset_id):
    from math import sin
    from simulation.presets import examples

    data = bars(
        tuple(
            100 + 20 * sin(i / 8) + 3 * sin(i) - (30 if i % 40 == 0 else 0)
            for i in range(180)
        )
    )
    versions = [p for p in examples() if p["template_id"] == preset_id]
    assert {p["language"] for p in versions} == {"python", "cpp"}
    outputs = [backtests.execute_strategy({**request(), **p}, data) for p in versions]
    assert outputs[0]["fills"], preset_id
    assert outputs[0]["fills"] == outputs[1]["fills"]
    assert outputs[0]["equity_curve"] == outputs[1]["equity_curve"]
    assert outputs[0]["metrics"] == outputs[1]["metrics"]
    if preset_id == "buy_and_hold":
        assert len(outputs[0]["fills"]) == 1
        assert (
            outputs[0]["metrics"]["total_return_percent"]
            == outputs[0]["metrics"]["benchmark_return_percent"]
        )


def test_breakout_uses_prior_channel_and_mean_reversion_exits():
    from simulation.presets import examples

    library = {p["template_id"]: p for p in examples() if p["language"] == "python"}
    breakout = backtests.execute_strategy(
        {
            **request(),
            **library["channel_breakout"],
            "params": {"entry_period": 3, "exit_period": 2},
            "warmup": 0,
        },
        bars((10, 11, 12, 14, 15, 8, 9)),
    )
    assert breakout["fills"][0]["signal_date"] == "2025-01-04"
    assert breakout["fills"][0]["time"] == "2025-01-05"
    assert breakout["fills"][-1]["side"] == "sell"
    reversion = backtests.execute_strategy(
        {
            **request(),
            **library["mean_reversion"],
            "params": {"lookback": 4, "entry_z": 1.5, "exit_z": 0},
            "warmup": 0,
        },
        bars((100, 100, 100, 50, 70, 100, 100)),
    )
    assert reversion["fills"][0]["signal_date"] == "2025-01-04"
    assert reversion["fills"][-1]["side"] == "sell"
    flat = backtests.execute_strategy(
        {**request(), **library["mean_reversion"], "warmup": 0}, bars((100,) * 40)
    )
    assert flat["fills"] == []


def wait_run(client, run_id):
    for _ in range(200):
        run = client.get("/api/backtests/" + run_id).json()
        if run["status"] not in ("queued", "running"):
            return run
        time.sleep(0.025)
    pytest.fail("Backtest did not finish")


def test_cache_persistence_invalidation_and_bypass(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", str(tmp_path / "cache.sqlite3"))
    calls = []
    data = bars()

    def history(*args):
        calls.append(args)
        return {"source": "demo", "bars": data}

    monkeypatch.setattr(market, "history", history)
    with TestClient(app) as client:
        first = client.post("/api/backtests", json=request()).json()
        assert wait_run(client, first["id"])["status"] == "completed"
        backtests.initialize()
        cached = client.post(
            "/api/backtests", json={**request(), "name": "Renamed"}
        ).json()
        assert cached["cache_hit"] and cached["id"] == first["id"]
        assert len(calls) == 1
        forced = client.post(
            "/api/backtests", json={**request(), "force_rerun": True}
        ).json()
        assert forced["id"] != first["id"] and not forced["cache_hit"]
        assert wait_run(client, forced["id"])["status"] == "completed"
        assert len(calls) == 2
        changed = client.post(
            "/api/backtests", json={**request(), "commission": 2}
        ).json()
        assert not changed["cache_hit"]
        assert wait_run(client, changed["id"])["status"] == "completed"
        # Expired snapshots fetch data again and updated OHLCV changes the result key.
        with database.connect() as db:
            db.execute("UPDATE backtest_cache SET expires=0")
        data[-1] = {**data[-1], "close": 41, "high": 42}
        refreshed = client.post("/api/backtests", json=request()).json()
        latest = wait_run(client, refreshed["id"])
        assert latest["result"]["data"]["sha256"] != cached["result"]["data"]["sha256"]


def test_native_engine_matches_reference():
    import random
    from simulation.native import build, simulate_native

    rng = random.Random(42)
    data = bars(tuple(rng.uniform(10, 100) for _ in range(150)))
    for warmup in (0, 20):
        for fee in (0, 2):

            def callback(ctx):
                return (None, 0, 0.3, 1)[len(ctx.history) % 4]

            settings = dict(
                initial_cash=10000, commission=fee, slippage_bps=15, warmup=warmup
            )
            reference = simulate(data, callback, **settings)
            native = simulate_native(data, {}, settings, build(), callback=callback)
            assert native["fills"] == reference["fills"]
            assert native["equity_curve"] == reference["equity_curve"]
            assert native["metrics"] == pytest.approx(reference["metrics"])


def test_history_pages_are_older_and_non_overlapping(monkeypatch):
    monkeypatch.setattr(market, "DATA_MODE", "demo")
    first = market.history("AAPL", 90)
    older = market.history("AAPL", 250, first["bars"][0]["time"])
    assert len(older["bars"]) == 250
    assert older["bars"][-1]["time"] < first["bars"][0]["time"]
    assert older["has_more"]


@pytest.mark.parametrize(
    "change",
    [
        {"code": "class Strategy:\n def on_bar(self, ctx): return None"},
        {"language": "cpp"},
        {"params": {"slow": 4, "fast": 2}},
        {"ticker": "MSFT"},
        {"start_date": "2025-02-01"},
        {"end_date": "2025-10-01"},
        {"initial_cash": 2000},
        {"commission": 2},
        {"slippage_bps": 10},
        {"warmup": 2},
    ],
)
def test_cache_key_covers_simulation_inputs(change):
    assert backtests.request_key(request()) != backtests.request_key(
        {**request(), **change}
    )


def test_pending_runs_coalesce_and_failures_are_not_cached(tmp_path, monkeypatch):
    import threading

    monkeypatch.setattr(database, "DATABASE_PATH", str(tmp_path / "pending.sqlite3"))
    entered, release = threading.Event(), threading.Event()

    def history(*args):
        entered.set()
        assert release.wait(5)
        raise ValueError("Data temporarily unavailable")

    monkeypatch.setattr(market, "history", history)
    with TestClient(app) as client:
        first = client.post("/api/backtests", json=request()).json()
        try:
            assert entered.wait(5)
            duplicate = client.post("/api/backtests", json=request()).json()
            assert duplicate["id"] == first["id"]
            assert not duplicate["cache_hit"]
        finally:
            release.set()
        assert wait_run(client, first["id"])["status"] == "failed"
        retry = client.post("/api/backtests", json=request()).json()
        assert retry["id"] != first["id"]
        assert wait_run(client, retry["id"])["status"] == "failed"
