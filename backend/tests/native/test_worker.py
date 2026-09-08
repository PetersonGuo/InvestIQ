"""Verify the native worker against an independent Python reference model."""

import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from datetime import date, timedelta
import pytest
from conftest import ROOT, wait_run
from test_http import query

spec = importlib.util.spec_from_file_location(
    "reference_engine", ROOT / "tests/reference_engine.py"
)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)


@pytest.mark.parametrize("warmup,fee", [(0, 0), (20, 2), (30, 1)])
def test_randomized_reference_parity(tmp_path, warmup, fee):
    rng = random.Random(42)
    bars = []
    for i in range(180):
        price = rng.uniform(10, 100)
        bars.append(
            {
                "time": (date(2025, 1, 1) + timedelta(days=i)).isoformat(),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 1000,
            }
        )
    settings = {
        "initial_cash": 10000,
        "commission": fee,
        "slippage_bps": 15,
        "warmup": warmup,
        "interval": "1d",
    }
    source = tmp_path / "strategy.cpp"
    source.write_text(
        '#include "strategy_api.h"\nextern "C" double on_bar(const SA_Bar*, int count, double, int64_t, const char*) { const double weights[] = {-1, 0, .3, 1}; return weights[count % 4]; }'
    )
    module = tmp_path / "strategy.so"
    flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            *flags,
            "-I",
            str(ROOT / "simulation"),
            str(source),
            "-o",
            str(module),
        ],
        check=True,
        capture_output=True,
    )
    payload = {
        "language": "cpp",
        "module_path": str(module),
        "bars": bars,
        "params": {},
        "settings": settings,
    }
    (tmp_path / "request.json").write_text(json.dumps(payload))
    subprocess.run(
        [
            str(ROOT / "build/native/stockassist-worker"),
            str(tmp_path / "request.json"),
            str(tmp_path / "result.json"),
        ],
        check=True,
        timeout=15,
        capture_output=True,
    )
    native = json.loads((tmp_path / "result.json").read_text())["result"]
    expected = reference.simulate(
        bars,
        lambda ctx: (None, 0, 0.3, 1)[len(ctx.history) % 4],
        **{k: v for k, v in settings.items() if k != "interval"}
    )
    assert native["equity_curve"] == expected["equity_curve"]
    assert native["fills"] == expected["fills"]
    assert native["metrics"] == pytest.approx(expected["metrics"])


@pytest.mark.parametrize(
    "language,code,detail",
    [
        (
            "python",
            'class Strategy:\n def on_bar(self, ctx): raise ValueError("strategy diagnostic")',
            "strategy diagnostic",
        ),
        (
            "python",
            "class Strategy:\n def on_bar(self, ctx): return True",
            "target weight",
        ),
        (
            "cpp",
            '#include "strategy_api.h"\n#include <cstdlib>\nextern "C" double on_bar(const SA_Bar*,int,double,int64_t,const char*) { std::abort(); }',
            "exited",
        ),
        (
            "cpp",
            '#include "strategy_api.h"\nextern "C" double on_bar(const SA_Bar*,int,double,int64_t,const char*) { volatile int x=0; while(true) { x=1; } }',
            "exited",
        ),
    ],
)
def test_failures_are_contained(client, language, code, detail):
    q = {**query(client, language), "code": code}
    submitted = client.post("/api/backtests", json=q).json()
    run = wait_run(client, submitted["id"])
    assert run["status"] == "failed"
    assert detail in run["error"]
    assert client.get("/health").json()["backend"] == "cpp"
