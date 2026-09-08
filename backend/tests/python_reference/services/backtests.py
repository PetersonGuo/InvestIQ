"""Persisted backtest jobs with bounded, separate strategy processes."""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from zoneinfo import ZoneInfo
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException
from core.database import connect
from services import market

ROOT = Path(__file__).resolve().parents[1]
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="backtest"
)
_slots = threading.BoundedSemaphore(2)
_submit_lock = threading.Lock()
CACHE_SECONDS = 3600


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def request_key(request):
    sources = (
        sorted((ROOT / "simulation").glob("*.py"))
        + sorted((ROOT / "simulation").glob("*.cpp"))
        + sorted((ROOT / "simulation").glob("*.h"))
        + [
            Path(__file__),
            ROOT / "services/market.py",
            ROOT / "services/ibkr.py",
            ROOT / "services/resolutions.py",
        ]
    )
    engine = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    inputs = {k: v for k, v in request.items() if k not in ("name", "force_rerun")}
    inputs.setdefault("interval", "1d")
    return digest([inputs, market.DATA_MODE, engine, sys.version, sys.platform])


def cache_day():
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def initialize():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS backtest_cache (
            key TEXT PRIMARY KEY, run_id TEXT NOT NULL, expires REAL NOT NULL, day TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS strategies (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, language TEXT NOT NULL,
            code TEXT NOT NULL, params_json TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT NOT NULL,
            request_json TEXT NOT NULL, result_json TEXT, error TEXT);
        """)
        db.execute(
            "UPDATE backtest_runs SET status='failed', error='Backend restarted before this run completed. Run it again.' WHERE status IN ('queued','running')"
        )


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def run_process(command, cwd, timeout, log):
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(ROOT),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "en_US.UTF-8",
    }
    with open(log, "wb") as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(
                f"Strategy process exceeded its {timeout}-second time limit."
            ) from exc
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    if code:
        detail = Path(log).read_text(errors="replace")[:4000]
        raise ValueError(f"Strategy process exited with code {code}. {detail}")


def execute_strategy(request, bars):
    from simulation.native import build

    engine_path = build()
    with tempfile.TemporaryDirectory(prefix="stockassist-strategy-") as folder:
        directory = Path(folder)
        language = request["language"]
        source = directory / ("strategy.py" if language == "python" else "strategy.cpp")
        source.write_text(request["code"])
        module_path = source
        if language == "cpp":
            compiler = (
                shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
            )
            if not compiler:
                raise ValueError(
                    "Install a C++17 compiler (Xcode Command Line Tools on macOS) to run C++ strategies."
                )
            shutil.copy(ROOT / "simulation" / "strategy_api.h", directory)
            module_path = directory / "strategy.so"
            flags = (
                ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
            )
            run_process(
                [
                    compiler,
                    "-std=c++17",
                    "-O2",
                    *flags,
                    str(source),
                    "-o",
                    str(module_path),
                ],
                directory,
                30,
                directory / "compile.log",
            )
        payload = {
            "engine_path": engine_path,
            "language": language,
            "module_path": str(module_path),
            "params": request["params"],
            "bars": bars,
            "settings": {
                key: request[key]
                for key in ("initial_cash", "commission", "slippage_bps", "warmup")
            },
        }
        input_path, output_path = directory / "request.json", directory / "result.json"
        input_path.write_text(json.dumps(payload, allow_nan=False))
        run_process(
            [
                sys.executable,
                "-m",
                "simulation.worker",
                str(input_path),
                str(output_path),
            ],
            directory,
            12,
            directory / "worker.log",
        )
        if not output_path.exists():
            raise ValueError("Strategy exited without producing a result.")
        response = json.loads(output_path.read_text())
        if "error" in response:
            raise ValueError(response["error"] + "\n" + response.get("traceback", ""))
        result = response["result"]
        result["logs"] = (directory / "worker.log").read_text(errors="replace")[:4000]
        return result


def load_bars(request):
    interval = request.get("interval", "1d")
    if interval == "1d":
        data = market.history(request["ticker"], 500)
        return data, [
            bar
            for bar in data["bars"]
            if request["start_date"] <= bar["time"] <= request["end_date"]
        ]
    from services.resolutions import utc_time

    start_time, end_time = utc_time(request["start_date"]), utc_time(
        request["end_date"]
    )
    before = request["end_date"]
    collected = {}
    for _ in range(40):
        data = market.history(request["ticker"], 500, before, interval)
        page = data["bars"]
        if not page:
            raise ValueError(
                "IBKR returned no bars before the requested start was reached. Choose a narrower window or a more recent start."
            )
        for bar in page:
            if start_time <= utc_time(bar["time"]) < end_time:
                collected[bar["time"]] = bar
        if len(collected) > 50000:
            raise ValueError(
                "Intraday backtests support up to 50,000 bars. Narrow the window or choose a larger interval."
            )
        oldest = page[0]["time"]
        if utc_time(oldest) >= utc_time(before):
            raise ValueError("Historical paging did not advance.")
        if utc_time(oldest) <= start_time:
            bars = [collected[k] for k in sorted(collected)]
            return data, bars
        before = oldest
    raise ValueError(
        "This window needs more than 40 historical pages. Choose a shorter window or larger interval."
    )


def _run(run_id, request, key):
    try:
        with connect() as db:
            db.execute(
                "UPDATE backtest_runs SET status='running' WHERE id=?", (run_id,)
            )
        data, bars = load_bars(request)
        if len(bars) < 2 or len(bars) <= request["warmup"] + 1:
            raise ValueError(
                "Not enough available bars in this date range after warmup."
            )
        data_key = digest([key, data["source"], bars])
        result = None
        if not request.get("force_rerun"):
            with connect() as db:
                cached = db.execute(
                    "SELECT r.result_json FROM backtest_cache c JOIN backtest_runs r ON r.id=c.run_id WHERE c.key=? AND r.status='completed'",
                    (data_key,),
                ).fetchone()
            if cached:
                result = json.loads(cached["result_json"])
        result = result if result is not None else execute_strategy(request, bars)
        result["engine_version"] = "2.1.0-cpp"
        result["interval"] = request.get("interval", "1d")
        if result["interval"] != "1d":
            result["metrics"]["sharpe_ratio"] = None
            result["metrics"]["annualized_return_percent"] = None
        result["data"] = {
            "ticker": request["ticker"],
            "source": data["source"],
            "start": bars[0]["time"],
            "end": bars[-1]["time"],
            "bar_count": len(bars),
            "sha256": hashlib.sha256(
                json.dumps(bars, sort_keys=True).encode()
            ).hexdigest(),
            "bars": bars,
        }
        result["assumptions"] = [
            "Daily regular-session bars; long-only, one symbol, whole shares, no leverage.",
            "Signals at close fill at the next available bar open, with fees and adverse slippage.",
            "Buy-and-hold benchmark enters on the first tradable open after warmup using the same costs.",
            "Open positions are marked to the final close; they are not forcibly sold.",
            "Sharpe uses 252 sessions/year and zero risk-free rate; win rate is per realized sell fill.",
            "No dividends, borrow, taxes, liquidity caps, or delisted-universe correction. Historical bars may be split-adjusted.",
            "Today’s scanner candidates are not a point-in-time historical universe. Avoid interpreting selection-biased results as expected returns.",
        ]
        if result["interval"] != "1d":
            result["assumptions"][
                0
            ] = f"{result['interval']} completed regular-session bars; long-only, one symbol, whole shares, no leverage. UTC timestamps; end time is exclusive."
            result["assumptions"][
                4
            ] = "Annualized return and Sharpe are omitted for intraday runs. Win rate is per realized sell fill."
        with connect() as db:
            db.execute(
                "UPDATE backtest_runs SET status='completed', result_json=? WHERE id=?",
                (json.dumps(result, allow_nan=False), run_id),
            )
            for cache_key in (key, data_key):
                db.execute(
                    "INSERT OR REPLACE INTO backtest_cache VALUES (?,?,?,?)",
                    (cache_key, run_id, time.time() + CACHE_SECONDS, cache_day()),
                )
    except Exception as exc:
        message = exc.detail if isinstance(exc, HTTPException) else str(exc)
        with connect() as db:
            db.execute(
                "UPDATE backtest_runs SET status='failed', error=? WHERE id=?",
                (message[:8000], run_id),
            )
    finally:
        _slots.release()


def submit(request):
    key = request_key(request)
    with _submit_lock:
        if not request.get("force_rerun"):
            with connect() as db:
                cached = db.execute(
                    "SELECT r.id,r.status FROM backtest_cache c JOIN backtest_runs r ON r.id=c.run_id WHERE c.key=? AND c.expires>? AND c.day=? AND r.status IN ('completed','queued','running')",
                    (key, time.time(), cache_day()),
                ).fetchone()
            if cached:
                response = get_run(cached["id"])
                response["cache_hit"] = cached["status"] == "completed"
                return response
        if not _slots.acquire(blocking=False):
            raise HTTPException(
                429, "Two backtests are already running. Wait for one to finish."
            )
        run_id = str(uuid4())
        try:
            with connect() as db:
                db.execute(
                    "INSERT INTO backtest_runs VALUES (?, ?, ?, ?, NULL, NULL)",
                    (
                        run_id,
                        "queued",
                        timestamp(),
                        json.dumps(request, allow_nan=False),
                    ),
                )
                db.execute(
                    "INSERT OR REPLACE INTO backtest_cache VALUES (?,?,?,?)",
                    (key, run_id, time.time() + CACHE_SECONDS, cache_day()),
                )
            _pool.submit(_run, run_id, request, key)
        except Exception:
            _slots.release()
            raise
        return {"id": run_id, "status": "queued", "cache_hit": False}


def get_run(run_id):
    with connect() as db:
        row = db.execute("SELECT * FROM backtest_runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Backtest not found.")
    data = dict(row)
    data["request"] = json.loads(data.pop("request_json"))
    data["result"] = (
        json.loads(data.pop("result_json")) if data["result_json"] else None
    )
    data.pop("result_json", None)
    return data
