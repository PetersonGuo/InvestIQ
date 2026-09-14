# Integration troubleshooting

[HTTP reference](api.md) · [Strategy authoring](strategies.md) · [Getting started](start.md)

## I do not know which program to write

Write an HTTP client if you want to control StockAssist from your own application. The working client is examples/api/run_backtest.py. Write a callback if you want StockAssist to evaluate your algorithm against historical bars. You can do both: your client reads a callback source file and submits it as code.

## The backend is healthy but data is missing

GET /health checks the server. GET /api/market/status checks the broker. Confirm that TWS/IB Gateway is logged in and its API port matches backend/.env. Live quote, history and news permissions are separate; check the exact endpoint's error or notes. The application does not substitute artificial data.

## My request returns 404, 403 or 415

For a terminal program use http://127.0.0.1:8000/api/...; health is http://127.0.0.1:8000/health. Send Content-Type: application/json with POST/PUT. Do not send an unrelated browser Origin header. Your browser frontend should use its same-origin proxy or one of the supported local origins.

## HTTP 202 returned but nothing has run yet

202 acknowledges a job; it is not the report. Keep id and poll GET /api/backtests/{id}. Read status on each response. completed has result; failed has error. A POST may return completed immediately when the result cache hits. A GET returning 200 can still describe a failed run.

## The same run appears again after I edit the name

Names do not invalidate cached results. Change code/parameters/settings or use force_rerun:true. External dependencies and files are not tracked in the cache. A cached run returns its original name and id. Do not use cached execution to test side effects or randomness.

## My C++ code compiles locally but the backtest cannot load it

One stock requires extern "C" double on_bar with the exact declared arguments. Two or more stocks require extern "C" int on_portfolio. The worker does not call main(). Include strategy_api.h rather than redefining its structs. The editor uploads one source file; it does not upload your build system, local headers or libraries. Compiler diagnostics appear in the failed run's error. Check syntax with c++ -std=c++17 -fsyntax-only -I backend/simulation YOUR_FILE.cpp.

## Python says history, bar or shares is missing

The single-stock context exposes ctx.history, ctx.bar and ctx.portfolio.shares. The portfolio context instead exposes ctx.histories[symbol] and ctx.portfolio.positions[symbol]. The number of requested stocks selects the interface, even if the source file was originally written for another mode.

## Python cannot import my library or open my file

Install imports into backend/.venv. Only source text is submitted; its original file directory is not the worker's current directory. Local module files beside your strategy are not automatically uploaded. For external inputs use an explicitly available path/package, understand the local execution permissions, and force a rerun when those inputs change.

## The strategy returns an invalid target

Single stock: Python None or C++ -1 means hold; otherwise return a finite number from 0 to 1. Booleans are invalid. Portfolio: Python returns None or a dict; C++ returns 0 or writes every weight and returns 1. The sum of absolute weights must fit max_gross_exposure. Portfolio targets are complete allocations, not incremental orders. Omitted Python symbols target zero.

## No fills or an unexpected first fill

Signals fill at the following open, not the signal close. Final-bar signals cannot fill. warmup=30 suppresses the first 30 bars' eligibility; the first eligible signal is at bar 31 and can fill at bar 32. Single-stock callbacks run during warmup, but portfolio callbacks start after it and receive warmup history. Do not mark your strategy permanently entered during discarded warmup calls. Also inspect fees, capital, whole-share sizing and unfilled_final_signal.

## Not enough shared bars or unexpected date coverage

The engine intersects every requested symbol with SPY and requires warmup+2 bars. Missing bars are not forward-filled. warmup is inside the requested range; it does not cause earlier data to be fetched. Daily jobs only look within the latest 500 available bars. Intraday jobs have limits of 31 days, 40 pages and 50000 bars per stock. Read result.data.start, end, bar_count and available_bars_before_alignment; choose a recent shorter interval/window or reduce warmup when appropriate.

## A long computation fails

The worker has a 12-second wall limit and 8-second CPU limit; C++ compilation has 30 seconds. Avoid refitting a large model or rescanning all history unnecessarily on every callback. Persist rolling state within the run or reduce the data window. Two active jobs trigger HTTP 429 for another submission. A client timeout does not cancel the server job.

## My alert exists but I do not get an email or an intraday trigger

Alerts evaluate completed daily closes approximately every 30 seconds while the backend runs. Notifications are persisted locally at /api/alerts/events and /api/pairs/events. Email/push delivery is not enabled. Repeating rules wait for a later observation to clear and cross back; staying beyond the threshold does not continuously notify. Pair rules expose last_error for stale, missing or invalid histories.

## My strategy can short but my paper order cannot

Those are separate engines. Multi-stock backtests support signed weights. The local paper account only buys/sells owned shares using real live quotes and IOC execution. Backtests model historical opens with configurable fixed fees/slippage; the paper account uses current bid/ask, displayed size and its own fee model. Neither sends brokerage orders.
