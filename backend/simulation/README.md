# Modular backtesting

Open `/research` in the web app. Scan IBKR stocks or search for a symbol, choose Python or C++, import/paste your code, configure dates and costs, and run. Saved strategy versions and completed runs persist in the current data-mode SQLite database. Export JSON includes the exact strategy source, parameters, bars, data checksum, assumptions, equity curve, and fills.

## Python strategy contract

A strategy file defines a no-argument `Strategy` class with `on_bar(self, ctx)`. Standard Python imports work using the backend virtual environment; install your own dependencies there. State stored on `self` persists across callbacks in a single run. A fresh worker/module/instance is created for every uncached run.

```python
class Strategy:
    def on_bar(self, ctx):
        lookback = int(ctx.params.get("lookback", 20))
        if len(ctx.history) < lookback:
            return None
        average = sum(b.close for b in ctx.history[-lookback:]) / lookback
        return 1.0 if ctx.bar.close > average else 0.0
```

Context fields:

- `ctx.history`: tuple of frozen `Bar` objects from the first input bar through the current completed bar, inclusive. Fields: `time` (ISO date for daily bars, UTC ISO timestamp for intraday bars), `open`, `high`, `low`, `close`, `volume`.
- `ctx.bar`: current completed bar.
- `ctx.portfolio`: frozen `cash`, `shares`, and marked-to-close `equity`.
- `ctx.params`: numeric parameters entered as JSON in the editor.

Return `None` to keep existing holdings without an order. Return a number in `[0, 1]` for a target invested fraction: `0` exits, `1` invests available cash, `0.5` targets half the portfolio. Invalid output fails the run with the date and error. Booleans are not weights.

## C++ strategy contract

Import a `.cpp` file or paste code in the C++ editor. A C++17 compiler is required for the engine in both languages (Xcode Command Line Tools on macOS, GCC/Clang on Linux). The worker compiles a shared library and loads it in a separate process through a narrow C ABI:

```cpp
#include "strategy_api.h"

extern "C" double on_bar(const SA_Bar* bars, int count, double cash,
                          int64_t shares, const char* params_json) {
    int lookback = static_cast<int>(sa_parameter(params_json, "lookback", 20));
    if (lookback < 1 || count < lookback) return -1.0;
    double mean = 0;
    for (int i = count - lookback; i < count; ++i) mean += bars[i].close;
    return bars[count - 1].close > mean / lookback ? 1.0 : 0.0;
}
```

`SA_Bar` has `int64_t timestamp` (bar start in UTC milliseconds; midnight for daily bars), then doubles `open`, `high`, `low`, `close`, `volume`. The supplied array contains `count` bars through the current bar; do not retain the pointer between calls. Static/global C++ state survives within a run and resets for the next process. `sa_parameter` reads flat numeric JSON parameters. Return `-1` to hold, otherwise `[0, 1]` as for Python.

The editor accepts one source file. Standard-library headers and the supplied `strategy_api.h` work out of the box. Python and C++ must implement their respective callback contracts; arbitrary existing programs with unrelated entry points need a small adapter. External C++ libraries requiring custom include/link flags are not configured by the editor.

The included moving-average examples in `examples/` implement the same algorithm in both languages. Automated tests compare their complete fills, equity curves, and metrics; both have also been checked on real IBKR daily bars.

## Engine and execution semantics

- One USD stock per run, selectable daily or intraday bars, whole shares, long-only, no leverage or shorting.
- At each bar's open, execute the target produced at the previous close. Apply adverse slippage to the open and a fixed commission per nonzero fill. Buy size is capped by available cash including the fee.
- At each close, mark holdings to market and call the strategy with history only through that close. Signals produced by the final bar cannot fill and are reported as unfilled.
- `warmup` suppresses trades for the first N bars while still calling the strategy to build state. The curve begins at the first bar after those N warmup bars; its first signal can fill on the following open.
- Buy-and-hold enters at that same first eligible open, with the same cash, fee, and slippage assumptions. It holds through the final close.
- Open positions are not forcibly liquidated at the end. The final equity includes their marked value.
- Equity drawdown is measured from the running equity peak, including initial capital. Sharpe uses close-to-close returns, a zero risk-free rate, and 252 sessions/year. Annualized return uses the number of simulated sessions and can be misleading for short windows. Win rate counts profitable realized sell fills, not grouped round trips.
- No dividend, tax, borrow, volume-participation, halt, spread, or full corporate-action simulation. There is no guaranteed next-open liquidity. Historical provider bars may be split-adjusted.
- Daily runs load up to 500 available bars and then filter the requested date range; the result reports the actual range. Today’s scanner candidates introduce historical selection/survivorship bias. This is a research tool, not a forecast.

## Modules

- `native_engine.cpp`: native event loop, fills, fees, positions, benchmark and drawdown.
- `../native/src/simulation.cpp`: native OHLCV validation, statistics, and result construction.
- `../native/src/worker.cpp`: isolated C++ worker; C++ strategies call their native ABI directly.
- `../native/src/python_adapter.cpp` and `python_strategy.py`: optional Python plugin and callback objects. Only Python runs load the interpreter.
- `../native/src/backtests.cpp`: native job scheduling, historical paging, compilation, persistence and cache.
- `../native/src/ibkr.cpp`: read-only native IBKR SDK adapter.
- `strategy_api.h`: version-one C ABI and parameter helper.

Both the server and worker are compiled ahead of time by `npm run build:backend`. The server and C++ worker do not link to Python; Python strategy support is a separate dynamically loaded plugin. The previous Python implementation is in `../tests/python_reference/` and is not part of runtime execution. Independent accounting parity tests use `../tests/reference_engine.py`.

## Local code execution

Run only strategy code you trust. These are ordinary Python/C++ programs with the local user's filesystem permissions, not security-sandboxed code. Workers receive a minimal environment rather than IBKR or Supabase environment variables, but that is not filesystem or network isolation. Do not expose this service publicly.

The API enforces local host/origin checks. There are at most two active jobs. Compilation has a 30-second wall limit; strategy execution has a 12-second wall limit and an 8-second CPU limit. Worker file/log output is capped by a 64 MiB file-size limit; the report keeps only the first 4,000 log characters. Process groups are terminated on completion or timeout. Native crashes, Python exceptions, and compiler errors are returned as failed runs rather than crashing the API. There is no portable memory sandbox or guarantee against malicious code.

## Included starter strategies

Choose a **Starter strategy**, select Python or C++, and click **Load preset**. Loading replaces the code, numeric parameters, and warmup; all remain editable. Both language versions implement the same rules.

| Preset | Default parameters | Warmup | Purpose |
| --- | --- | ---: | --- |
| Moving average crossover | `fast=10`, `slow=30` | 30 | Trend-following baseline; invests while the fast closing-price average exceeds the slow average. |
| Buy and hold | None | 0 | Passive baseline; invests once at the first eligible open and holds. Matches the report benchmark with identical costs and warmup. |
| Price-channel breakout | `entry_period=20`, `exit_period=10` | 20 | Buys a close above prior highs; exits below prior lows. Excludes the current bar from both channels. |
| Z-score mean reversion | `lookback=20`, `entry_z=2`, `exit_z=0` | 20 | Buys unusually low closes relative to their rolling mean and exits on recovery. Uses population standard deviation including the current completed close; flat windows hold. |

These are transparent comparison baselines, not tuned or proven-profitable strategies. Trend and breakout rules can whipsaw; mean reversion can stay exposed during sustained declines. Use the same data window, costs, and warmup to compare strategies on equal terms. Prefer a separate evaluation window after choosing parameters rather than optimizing and judging on the same bars.

## Persistent result cache

Identical requests return a completed result immediately, including after backend restarts. Keys cover strategy source/language, parameters, ticker/dates, costs/warmup, provider, compiler/Python build versions, and native engine/adapter source fingerprints. Display names do not invalidate results. Concurrent identical requests share a pending job. Failed runs are never reused.

A market-data snapshot is reused for up to one hour and expires at the next New York calendar date. After expiration the provider is checked again; an identical bars checksum permits reuse of the saved computation. Data corrections therefore invalidate results on the next snapshot refresh. The provider itself may cache responses for 60 seconds.

Select **Force rerun (skip result cache)** or send `force_rerun: true` to `POST /api/backtests` to execute again and request the current provider snapshot. Use this for random strategies, external files/imports, changed dependencies, network inputs, or any side effects: these are not fingerprinted. Cached runs retain their original name and run ID. The native engine is built ahead of time under `backend/build/native`; execution remains in a separate worker.

## Intraday bars and trade records

Charts and the strategy lab offer 1/5/10/15/30-second, 1/2/3/5/15/30-minute, 1/4-hour, and daily candles. One second is the finest IBKR historical candle size. Intraday timestamps and lab datetime inputs are UTC. Only completed bars from regular sessions are used. Changing the lab interval selects the latest available page as a starting window; edit its start/end to request more history. Intraday end times are exclusive. Indicator lookbacks remain counts of bars.

Intraday jobs fetch the selected range across up to 40 bounded pages, limited to 50,000 bars and 31 calendar days. Oversized or unavailable ranges fail explicitly instead of silently running a shortened window. The C++ engine and Python strategy contract accept intraday timestamps without converting data to daily bars. Annualized return and Sharpe are currently omitted for intraday runs; total return, benchmark, fills and drawdown remain available.

IBKR history pages follow duration/bar-size limits. The adapter caches pages for 60 seconds, spaces small-bar requests, prevents identical small requests within 15 seconds, and keeps a conservative local budget of 55 small-bar/tick requests per ten minutes. IBKR can impose additional account-level limits. Bars of 30 seconds and smaller have roughly six months of retention. Panning at a retention/permission limit shows the provider error; choose a coarser interval to go farther back.

The dashboard **Individual trades · IBKR Time & Sales** panel fetches the latest regular-session historical trade records and exports JSON. Every returned trade is preserved, including multiple trades at the same one-second timestamp. IBKR may return more than 1,000 records to complete a second. This is a cached snapshot, not a streaming subscription. The backtester consumes candles, not these raw trade records. Daily alerts and paper fills still use completed daily closes.

Sources: [Historical bars](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-bars/receiving-historical-bars), [request pacing](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-data-limitations/pacing-violations-for-small-bars-30-secs-or-less), [historical Time & Sales](https://interactivebrokers.github.io/tws-api/historical_time_and_sales.html).
