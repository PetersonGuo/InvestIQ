#!/usr/bin/env python3
"""Submit a local Python/C++ strategy to StockAssist and save the finished report.

Run from the repository root. This client uses Python's standard library only.
It requests provider data through StockAssist; it never submits paper orders.
"""
import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def api(base, path, body=None):
    request = Request(
        base.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            return json.load(response)
    except HTTPError as error:
        message = error.read().decode(errors="replace")
        try:
            message = json.loads(message).get("detail", message)
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"HTTP {error.code}: {message}") from error
    except URLError as error:
        raise RuntimeError(f"Cannot reach StockAssist at {base}: {error.reason}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="One .py or .cpp strategy file")
    parser.add_argument("--tickers", nargs="+", default=["AAPL"], help="1 symbol for on_bar; 2–10 for a portfolio")
    parser.add_argument("--start", help="Daily date or intraday UTC timestamp")
    parser.add_argument("--end", help="Daily inclusive date or intraday exclusive UTC timestamp")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--params", default="{}", help="JSON object containing numeric values")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--cash", type=float, default=100000)
    parser.add_argument("--commission", type=float, default=1)
    parser.add_argument("--slippage-bps", type=float, default=5)
    parser.add_argument("--max-gross-exposure", type=float, default=1)
    parser.add_argument("--borrow-rate-percent", type=float, default=3)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=600, help="Polling deadline in seconds; does not cancel the server job")
    parser.add_argument("--output", type=Path, default=Path("backtest-result.json"))
    args = parser.parse_args()
    language = {".py": "python", ".cpp": "cpp"}.get(args.source.suffix.lower())
    if not language:
        parser.error("--source must end in .py or .cpp")
    if args.interval != "1d" and (not args.start or not args.end):
        parser.error("Intraday runs require explicit --start and --end UTC timestamps")
    try:
        params = json.loads(args.params)
        code = args.source.read_text()
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not isinstance(params, dict):
        parser.error("--params must be a JSON object")
    payload = {
        "name": args.source.stem, "language": language, "code": code,
        "ticker": args.tickers[0], "tickers": args.tickers, "params": params,
        "interval": args.interval,
        "start_date": args.start or str(date.today() - timedelta(days=180)),
        "end_date": args.end or str(date.today()),
        "initial_cash": args.cash, "commission": args.commission,
        "slippage_bps": args.slippage_bps, "warmup": args.warmup,
        "max_gross_exposure": args.max_gross_exposure,
        "borrow_rate_percent": args.borrow_rate_percent,
        "force_rerun": args.force_rerun,
    }
    try:
        health = api(args.base, "/health")
        print(f"Backend: {health['backend']} · provider: {health['data_mode']}", flush=True)
        run = api(args.base, "/api/backtests", payload)
        run_id = run["id"]
        print(f"Run: {run_id} · {run['status']} · cache hit: {run.get('cache_hit', False)}", flush=True)
        deadline = time.monotonic() + args.timeout
        while run["status"] in ("queued", "running"):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Polling timed out. The job may still run; GET /api/backtests/{run_id} to resume.")
            time.sleep(1)
            run = api(args.base, f"/api/backtests/{run_id}")
        if run["status"] != "completed":
            raise RuntimeError(f"Run {run_id} failed: {run.get('error', 'Unknown error')}")
        args.output.write_text(json.dumps(run, indent=2) + "\n")
        result = run["result"]
        print("Actual data:", result["data"]["start"], "through", result["data"]["end"], flush=True)
        print(json.dumps(result["risk_comparison"], indent=2))
        print(f"Full request, code, bars, fills and report saved to {args.output}")
    except (RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
