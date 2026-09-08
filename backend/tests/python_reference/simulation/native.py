"""ctypes bridge: a single native event loop; Python callbacks remain supported."""

import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from types import MappingProxyType
from simulation.engine import Bar, Context, Portfolio, summarize
from math import isfinite

ROOT = Path(__file__).resolve().parent


class NativeBar(c.Structure):
    _fields_ = [("timestamp", c.c_int64)] + [
        (name, c.c_double) for name in ("open", "high", "low", "close", "volume")
    ]


class Point(c.Structure):
    _fields_ = [
        (name, c.c_double) for name in ("equity", "benchmark", "drawdown", "cash")
    ] + [("shares", c.c_int64)]


class Fill(c.Structure):
    _fields_ = [(name, c.c_int) for name in ("index", "signal", "side")] + [
        ("quantity", c.c_int64),
        ("price", c.c_double),
        ("pnl", c.c_double),
    ]


Callback = c.CFUNCTYPE(
    c.c_double, c.POINTER(NativeBar), c.c_int, c.c_double, c.c_int64, c.c_char_p
)


def build():
    compiler = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    if not compiler:
        raise ValueError("Install a C++17 compiler to build the backtest engine.")
    digest = hashlib.sha256(
        (ROOT / "native_engine.cpp").read_bytes()
        + (ROOT / "strategy_api.h").read_bytes()
        + sys.platform.encode()
    ).hexdigest()
    directory = ROOT.parent / "data" / "native"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (digest + ".so")
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            output = Path(temporary) / "engine.so"
            flags = (
                ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
            )
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O3",
                    *flags,
                    str(ROOT / "native_engine.cpp"),
                    "-o",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode:
                raise ValueError(
                    "Native engine compilation failed: " + result.stderr[:4000]
                )
            os.replace(output, target)
    return str(target)


def simulate_native(
    bars, params, settings, engine_path, callback=None, strategy_path=None
):
    initial_cash, commission, slip, warmup = (
        settings[k] for k in ("initial_cash", "commission", "slippage_bps", "warmup")
    )
    if len(bars) < 2 or not 0 <= warmup <= len(bars) - 2:
        raise ValueError("Choose at least two trading bars after warmup.")
    if (
        not isfinite(initial_cash)
        or initial_cash <= 0
        or not 0 <= commission <= initial_cash
        or not 0 <= slip <= 1000
    ):
        raise ValueError("Invalid cash, commission or slippage.")
    history = tuple(Bar(**row) for row in bars)
    for i, bar in enumerate(history):
        values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
        if (
            not all(isfinite(v) for v in values)
            or min(values[:4]) <= 0
            or bar.volume < 0
        ):
            raise ValueError("Invalid OHLCV data.")
        if bar.low > min(bar.open, bar.close) or bar.high < max(bar.open, bar.close):
            raise ValueError("Invalid OHLC range.")
        if i and bar.time <= history[i - 1].time:
            raise ValueError("Bars must have unique, ascending dates.")
    data = (NativeBar * len(bars))(
        *[
            NativeBar(
                int(
                    datetime.fromisoformat(b.time)
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                    * 1000
                ),
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume,
            )
            for b in history
        ]
    )
    errors = []
    parameters = MappingProxyType(params)
    if strategy_path:
        strategy = c.CDLL(strategy_path)
        function = Callback(("on_bar", strategy))
    else:

        @Callback
        def function(_bars, count, cash, shares, _params):
            try:
                value = callback(
                    Context(
                        history[:count],
                        Portfolio(
                            cash, shares, cash + shares * history[count - 1].close
                        ),
                        parameters,
                    )
                )
                if value is None:
                    return -1
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError(
                        f"Invalid strategy target weight on {history[count-1].time}; expected None or 0 to 1."
                    )
                return value
            except BaseException as exc:
                errors.append(exc)
                return float("nan")

    engine = c.CDLL(engine_path)
    execute = engine.sa_simulate
    execute.argtypes = [
        c.POINTER(NativeBar),
        c.c_int,
        Callback,
        c.c_char_p,
        c.c_double,
        c.c_double,
        c.c_double,
        c.c_int,
        c.POINTER(Point),
        c.POINTER(Fill),
        c.POINTER(c.c_int),
        c.POINTER(c.c_double),
        c.c_char_p,
    ]
    execute.restype = c.c_int
    points, fills = (Point * len(bars))(), (Fill * len(bars))()
    count, pending, error = c.c_int(), c.c_double(), c.create_string_buffer(512)
    status = execute(
        data,
        len(bars),
        function,
        json.dumps(params).encode(),
        initial_cash,
        commission,
        slip,
        warmup,
        points,
        fills,
        c.byref(count),
        c.byref(pending),
        error,
    )
    if errors:
        raise errors[0]
    if status:
        raise ValueError(error.value.decode())
    curve = [
        {
            "time": history[i].time,
            "equity": round(p.equity, 6),
            "benchmark": round(p.benchmark, 6),
            "drawdown_percent": round(p.drawdown, 6),
            "cash": round(p.cash, 6),
            "shares": p.shares,
        }
        for i, p in enumerate(points)
        if i >= warmup
    ]
    trades = [
        {
            "signal_date": history[f.signal].time,
            "time": history[f.index].time,
            "side": "buy" if f.side == 1 else "sell",
            "quantity": f.quantity,
            "price": round(f.price, 6),
            "commission": commission,
            "realized_pnl": None if f.side == 1 else round(f.pnl, 6),
        }
        for f in fills[: count.value]
    ]
    closed = [f.pnl for f in fills[: count.value] if f.side == -1]
    return summarize(
        curve,
        trades,
        closed,
        initial_cash,
        min(p.drawdown for p in points[warmup:]),
        commission * count.value,
        points[-1].shares,
        None if pending.value == -1 else (pending.value, ""),
    )
