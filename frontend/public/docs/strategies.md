# Write a strategy

[Start here](start.md) if you need to launch StockAssist. Use [the HTTP guide](api.md) to submit the code from another program.

## Select the right callback

| Selected stocks | Python entry point | C++ entry point | Valid result |
| --- | --- | --- | --- |
| One | Strategy.on_bar(ctx) | on_bar | None / -1 to hold, or a target fraction from 0 to 1 |
| Two through ten | Strategy.on_bar(ctx) with portfolio context | on_portfolio | None / return 0 to hold, or complete signed portfolio targets |

StockAssist loads one source file per run. It does not run a C++ main() function or a Python command-line entry point. Adapt your algorithm by calling it from the required callback. The number of requested tickers determines the context and C++ symbol the worker loads.

## Single-stock Python

Save this as my_strategy.py. No StockAssist Python package needs to be imported.

```python
class Strategy:
    def on_bar(self, ctx):
        period = int(ctx.params.get("period", 20))
        if period < 1:
            raise ValueError("period must be positive")
        if len(ctx.history) < period:
            return None
        mean = sum(bar.close for bar in ctx.history[-period:]) / period
        return 1.0 if ctx.bar.close > mean else 0.0
```

```sh
python3 examples/api/run_backtest.py \
  --source my_strategy.py --tickers AAPL \
  --params '{"period":20}' --warmup 20 --output my-python-result.json
```

| Field | Type and meaning |
| --- | --- |
| ctx.history | Tuple of completed Bar objects through the current close, inclusive |
| ctx.bar | The last Bar in ctx.history |
| bar.time | Daily YYYY-MM-DD or intraday UTC ISO timestamp |
| bar.open / high / low / close / volume | Numeric OHLCV fields |
| ctx.portfolio.cash | Cash in dollars |
| ctx.portfolio.shares | Whole-share position; nonnegative in a single-stock run |
| ctx.portfolio.equity | Cash plus shares marked at the current close, in dollars |
| ctx.params | Read-only mapping of flat numeric parameters |

Return None to leave holdings unchanged. Return 0.0 to target no holdings, 1.0 to invest the account, or 0.5 to target approximately half the account's equity. These are target weights, not share counts or buy/sell commands. True, False, NaN and infinity are invalid. Repeating 0.5 can cause rebalancing as prices change; returning None avoids that instruction.

State on self persists within one uncached run. Keep only data you need. A new run gets a fresh Strategy instance. Read the provided history rather than fetching data inside on_bar; callbacks should compute quickly and must not use future observations or private adapter internals.

## Single-stock C++

Save this as my_strategy.cpp. The backend supplies strategy_api.h when it compiles your source.

```cpp
#include "strategy_api.h"

extern "C" double on_bar(const SA_Bar* bars, int count,
                          double cash, int64_t shares,
                          const char* params_json) {
    int period = static_cast<int>(sa_parameter(params_json, "period", 20));
    if (period < 1 || count < period) return -1.0;
    double mean = 0.0;
    for (int i = count - period; i < count; ++i) mean += bars[i].close;
    mean /= period;
    return bars[count - 1].close > mean ? 1.0 : 0.0;
}
```

```sh
python3 examples/api/run_backtest.py \
  --source my_strategy.cpp --tickers AAPL \
  --params '{"period":20}' --warmup 20 --output my-cpp-result.json
```

The Python command above is an HTTP client; your algorithm and simulation run in C++. You can replace that client with curl, C++, or any other HTTP client.

| Argument | Meaning |
| --- | --- |
| bars | Array of completed SA_Bar records; read only indices 0 through count-1 |
| count | Number of visible bars including the current close |
| cash | Cash in dollars |
| shares | Current whole-share holdings |
| params_json | Flat numeric parameter JSON; sa_parameter supplies a fallback for missing keys |

SA_Bar contains int64_t timestamp in UTC epoch milliseconds, followed by double open, high, low, close and volume. Daily bars use midnight UTC as their date label. It is not a daily execution timestamp. To compute current equity, use cash + shares * bars[count-1].close.

Return exactly -1.0 to hold, or a finite value in [0,1] to set a target. Other negative values fail. extern "C" is required so the worker can find the exported function. Do not retain pointers to bars, strings or output buffers after the callback returns. Static/global state can persist within the worker process for that run.

## Multi-stock Python

The portfolio context has histories and positions instead of history, bar and shares. History values support len(), integer indexing, negative indexing and slicing. All symbols have the same visible timestamps and length. Dict iteration follows the submitted ticker order.

```python
class Strategy:
    def on_bar(self, ctx):
        if any(ctx.portfolio.positions.values()):
            return None
        weight = 1.0 / len(ctx.histories)
        return {symbol: weight for symbol in ctx.histories}
```

| Field | Meaning |
| --- | --- |
| ctx.histories[symbol] | Completed history for that requested symbol; [-1] is its current bar |
| ctx.portfolio.positions[symbol] | Signed whole-share position; negative means short |
| ctx.portfolio.cash / equity | Shared cash and marked account equity in dollars |
| ctx.params | Numeric strategy parameters |

Return None to hold every position. A dictionary is a complete new portfolio target: omitted symbols target zero. Unknown symbols and boolean/non-numeric weights fail. For example, {"AAPL":0.5,"MSFT":-0.5} means long exposure of 50% and short exposure of 50% of equity; gross exposure is 1.0 and target net exposure is 0.0. This illustrates weights, not a recommended trade. Return {"AAPL":0.0,"MSFT":0.0} to flatten both legs. An empty dictionary also targets zero for all symbols.

## Multi-stock C++

Save this as my_basket.cpp and submit at least two tickers.

```cpp
#include "strategy_api.h"

extern "C" int on_portfolio(const SA_Asset* assets, int asset_count,
                            int count, double cash, double equity,
                            const char* params_json, double* target_weights) {
    for (int i = 0; i < asset_count; ++i)
        if (assets[i].shares != 0) return 0;
    for (int i = 0; i < asset_count; ++i)
        target_weights[i] = 1.0 / asset_count;
    return 1;
}
```

```sh
python3 examples/api/run_backtest.py \
  --source my_basket.cpp --tickers AAPL MSFT NVDA \
  --warmup 0 --output my-basket-result.json
```

Each SA_Asset contains const char* symbol, const SA_Bar* bars and int64_t shares. The assets array follows request ticker order. Read the current close as assets[i].bars[count-1].close. Return 0 to leave holdings alone. To rebalance, write every target_weights[i] and return 1. The output array is not your persistent state. Positive weights mean long, negative weights mean short; the sum of absolute targets cannot exceed max_gross_exposure. Its default is 1.0 and maximum is 2.0.

The complete declarations live in backend/simulation/strategy_api.h. Standard C++17 headers, strategy_api.h and pair_math.h are supplied. Other source/header files are not uploaded with your file, and custom compiler include paths or linker flags are not supported by this API. For a standalone syntax check:

```sh
c++ -std=c++17 -fsyntax-only -I backend/simulation my_basket.cpp
```

## Execution timing and warmup

The callback sees a completed close. The resulting target fills at the next available bar's open, with adverse slippage and commission. It never fills at the same close. Shares are rounded to whole numbers. Final-bar signals cannot fill; inspect unfilled_final_signal in the result. Positions still open at the end are marked to the final close.

With warmup=0, the first callback sees bar 1 and its signal can fill at bar 2's open. With warmup=30, the first eligible decision sees bar 31 and can fill at bar 32's open. SPY and stock/basket buy-and-hold references enter at the same eligible open.

There is an important current difference between the two engines:

| Mode | Warmup callbacks |
| --- | --- |
| Single stock | Called on every input bar, including warmup; warmup targets are discarded |
| Portfolio | First callback occurs after warmup and sees the full prefix of warmup history |

For portable stateful algorithms, initialize rolling state from the visible history on the first callback. Do not assume a portfolio callback ran on each warmup bar. warmup does not fetch extra observations before start_date; those bars come out of the requested/aligned data window. At least warmup + 2 aligned bars are required.

SPY and every selected stock are intersected on exact timestamps; missing bars are never forward-filled. SPY is an engine reference, not automatically an extra symbol in your callback. Add SPY to tickers only if your algorithm should observe or trade it.

## Costs, risk and limitations

Single-stock strategies are long-only without leverage. Portfolio strategies allow signed targets and charge calendar-time borrow on short positions using borrow_rate_percent. Market moves can change exposure after sizing. There is no borrow-availability, maintenance-margin liquidation, dividend, tax, market-impact or intrabar risk model. Backtest fills do not use live bid/ask quotes or displayed liquidity; the separate paper account does.

risk_comparison compares strategy, SPY and stock/basket buy-and-hold on the same dates. SPY is a price-return ETF proxy, not a dividend-reinvested S&P 500 index. Daily volatility/ratios use 252 observations per year; intraday risk is per-bar. Undefined ratios are null. Read risk_basis and assumptions in each result before comparing runs.

## Dependencies, debugging and caching

Python imports use backend/.venv. Install dependencies there, for example with backend/.venv/bin/python -m pip install PACKAGE. The worker runs in a temporary directory; relative paths are not resolved against your original strategy file. Only the source text is submitted. Package dependencies and external files are not bundled.

print() and C++ stdout/stderr appear in result.logs, capped at 4,000 characters. A missing entry point, invalid target, exception, compiler failure or timeout appears in the failed run's error field. GET requests can succeed with HTTP 200 while the run itself has status failed.

Identical requests may return a cached completed run without executing your code. Use --force-rerun or force_rerun:true when debugging, using randomness, reading external files, or changing imports/dependencies. Source, parameters and engine changes affect the cache; external file contents and installed package changes do not.

C++ compilation has a 30-second wall limit. Execution has a 12-second wall limit and an 8-second CPU limit. Two backtests may be active at once. Run only trusted code: workers have local filesystem permissions and are not security sandboxes.

## Exposure-adjusted C++ strategies

In Pairs, **Save C++ backtest strategy** exports the selected economic relationship, factor proxies and parameters to Research. The new header `relative_value_math.h` is supplied automatically when compiling C++ callbacks. Six configured non-event starters appear when C++ is selected; event exports require sourced, dated cash/share terms.

The callback trades the target, reference basket and every selected market/sector/factor hedge. It refits before the signal bar and rebalances at the following open; it never trades just a residual label while omitting the hedge legs. The shared model supports up to 20 reference constituents plus eight exposure hedges. Portfolio jobs accept up to 32 total symbols. Warmup accepts 0–1000 bars; an exported model requests lookback + 2 and independently refuses to estimate before it has sufficient history.

See [relative-value model details](/docs/api#relative-value-models) for equations, invalid-model behavior, proxy limitations and daily lead-lag timing. Hedge neutrality is in sample; use the existing same-period SPY comparison and portfolio risk metrics to evaluate realized outcomes after costs.

## Cost-aware relative-value C++ strategies

New relative-value presets and exports enable cost_aware=1. Existing saved source versions and completed runs do not change; load a fresh preset or use **Backtest this setup** to run the updated source. Current price-monitor entry labels do not incorporate this backtest cost policy.

The entry test compares a conservative capture scenario with an estimated round-trip cost. For mean reversion, capture_fraction times (absolute score minus exit_z) times residual-path standard deviation is divided by the hedge's pre-normalization gross exposure. For daily lead-lag, the absolute next-day residual forecast replaces the distance to the exit band. This is a scenario, not a calibrated expected profit or a probability of convergence.

Round-trip cost includes slippage on entry and exit gross notional, commissions for both sides of each nonzero leg, and short borrow over cost_holding_days calendar days. Entry requires the capture scenario to exceed cost_cushion times this estimate. The C++ callback reads actual engine settings through _sa_commission, _sa_slippage_bps, _sa_borrow_rate_percent and _sa_max_gross_exposure; user parameter values cannot override those settings.

| Parameter | Default | Meaning |
| --- | ---: | --- |
| cost_aware | 1 | Set 0 only for comparison with the previous trading policy |
| cost_cushion | 2 | Required multiple of estimated costs; minimum 1 |
| capture_fraction | 0.5 | Fraction of modeled move assumed captured; greater than 0 and at most 1 |
| cost_holding_days | 10 | Calendar-day borrow assumption, 1–365; actual holding time may differ |
| rebalance_band | 0.05 | Skip small total absolute equity-weight changes |
| max_weight_drift | 0.15 | Force correction above this total absolute weight drift, even when costs are unfavorable |
| max_holding_bars | 20 | Close after this many daily signal bars from the entry decision; 1–250 |

For moderate hedge drift, the policy compares capture scenario times turnover with estimated incremental rebalance fees/slippage. Large drift overrides this saving rule. Exit-band crossings, stop-band crossings, signal reversal, expired/inconsistent event terms and model failure still close positions. Targets below 1× allowed gross exposure are scaled to the configured cap. Whole-share rounding and next-open prices can make actual costs/weights differ from the signal-close estimates.

Research results show total_slippage_cost, commissions and short borrow separately. The optional strategy_diagnostics object records entry_checks, entries_rejected_by_cost, rebalances_skipped, risk_rebalances and time_exits. A custom C++ strategy may export `extern "C" const char *sa_strategy_diagnostics()` returning valid JSON after the run; the returned string must remain valid when read by the worker. Diagnostics do not alter execution.

A no-trade result can avoid losses but does not demonstrate profitable alpha. Historical backtests omit dividends, borrow availability and some market frictions. These rules need independent forward validation; they do not add the fundamental valuation model to historical backtests.
