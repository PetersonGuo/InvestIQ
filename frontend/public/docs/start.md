# Build with StockAssist

StockAssist has two APIs. Choose the one that matches what you want to build.

| Your goal | Interface | Your entry point |
| --- | --- | --- |
| Write a script or application that fetches data, manages alerts, or runs backtests | Local HTTP JSON API | Send requests to http://127.0.0.1:8000 |
| Write an algorithm that the backtester evaluates | Strategy callback API | Python Strategy.on_bar or C++ on_bar / on_portfolio |

Your application → HTTP API → real market data → C++ backtest worker → your strategy callback → fills, benchmarks and risk report.

The strategy decides target holdings. The engine manages the historical clock, fills, positions and reporting. A backtest never submits a paper or brokerage order. The paper account has a separate HTTP endpoint.

## Start the services

Run these commands from the StockAssist repository root:

```sh
npm run setup
npm run dev
```

Setup installs the project dependencies and builds the native backend. An installed IBKR C++ SDK and native compiler dependencies are required; see backend/README.md in the repository for those prerequisites. Use the existing configured backend/.env, or copy backend/.env.example if you do not have one. The default provider is IBKR. Open TWS or IB Gateway, log in, enable socket API access, and make its port match IBKR_PORT in backend/.env. IBKR_HOST defaults to 127.0.0.1 and IBKR_CLIENT_ID to 71. Keep the broker API read-only: StockAssist does not route brokerage orders.

The frontend runs on port 3000; the backend runs on port 8000. The supported providers are ibkr and massive. Price data is never fabricated when a provider is unavailable. Provider permissions for historical bars, live quotes and news can differ.

```sh
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/api/market/status
curl --fail-with-body 'http://127.0.0.1:8000/api/stocks/AAPL?days=90&interval=1d'
```

A healthy backend does not imply that TWS is connected. Check connected and last_error in market/status. Historical-data access also does not guarantee a live quote subscription.

## Run your first strategy

The repository includes complete Python and C++ strategies under backend/simulation/examples. These are algorithms that run on provider data. To run the moving-average strategy from your own terminal program:

```sh
python3 examples/api/run_backtest.py \
  --source backend/simulation/examples/moving_average.cpp \
  --tickers AAPL --params '{"fast":10,"slow":30}' \
  --warmup 30 --output moving-average-result.json
```

Switch the source extension to .py to run the Python version. The client defaults to a daily window covering the last 180 calendar days. It submits the source, polls the asynchronous job, prints risk comparisons and writes the complete JSON response. Its only dependencies are Python's standard library and a running StockAssist backend. The output file is overwritten if it already exists.

For multiple stocks, use a portfolio strategy:

```sh
python3 examples/api/run_backtest.py \
  --source backend/simulation/examples/equal_weight.cpp \
  --tickers AAPL MSFT NVDA --warmup 0 \
  --output basket-result.json
```

For the included cointegration strategy, use pairs_mean_reversion.cpp or pairs_mean_reversion.py, exactly two tickers, warmup 60, and parameters {"lookback":60,"entry_z":2,"exit_z":0.5,"stop_z":4}. This strategy selects spread directions itself; the HTTP pair-alert system is independent.

## Write your own program

[The HTTP API guide](api.md) explains request bodies, responses, errors, caching and polling. [The strategy guide](strategies.md) gives complete callback examples, field definitions and execution timing. [Troubleshooting](troubleshooting.md) covers common compiler, data and integration failures.

You can also use Research in the app: choose a language, import one .py or .cpp source file, select stocks and settings, then run it. A saved strategy version is a source-code record, not a running program. Loading a preset replaces the code and its settings in the editor; saving creates a new version.

## What runs where

| Component | Responsibility |
| --- | --- |
| Your HTTP client | Sends JSON requests, polls jobs, reads results; may be written in any language |
| C++ backend | Fetches IBKR/Massive data, stores rules and jobs in SQLite, schedules workers |
| C++ worker | Compiles/loads a strategy, simulates execution, produces benchmark and risk reports |
| Python adapter | Loads only for Python strategies; exposes the same completed-bar workflow |
| Frontend | Displays data and reports through its same-origin API proxy |

There is no required HTTP API key for this local single-user service. Broker/provider credentials stay in backend configuration. Bind the service to loopback. Strategy code runs with the local user's permissions; worker timeouts contain ordinary crashes and hangs, but do not provide a security sandbox.

## Use a C++ HTTP client

The separate examples/api/run_backtest.cpp program uses main() because it is an external client. Your submitted .cpp strategy still uses on_bar or on_portfolio. It uses the project's existing httplib and JSON headers, with no Python runtime:

```sh
c++ -std=c++17 -O2 -pthread -I backend/native/vendor \
  examples/api/run_backtest.cpp -o /tmp/stockassist-client
/tmp/stockassist-client backend/simulation/examples/moving_average.cpp \
  AAPL 2026-06-01 2026-09-01 cpp-client-result.json
```

Select dates within your provider's available history. Supply comma-separated symbols for a portfolio, such as AAPL,MSFT with equal_weight.cpp. An optional final argument selects the backend port. This compact client uses daily bars, warmup 30 and the standard costs; edit its payload to set parameters or other settings. It polls for up to ten minutes and writes the complete report, overwriting the named output file if present. Both client examples submit backtests only, never paper orders.

## Find pairs automatically

Open **Pairs → Advanced strategy setup → Automatic pair discovery**. Choose an IBKR scanner that refreshes its stock list every run, or enter your own 2–20 symbols. Set the repeat interval, correlation and cointegration filters, entry z-score, and maximum managed alerts, then select **Enable automatic discovery**. The first scan starts shortly; later scans add qualifying pairs without replacing your existing rules. Each generated alert includes a fitted hedge ratio and signal directions and remains active after firing.

Keep the backend running and TWS/IB Gateway connected. The schedule, next run, errors and alerts survive restarts. The browser may be closed. Runs use completed daily bars; the interval is measured from completion, not a fixed market-close time. Use **Run now**, **Pause schedule**, or **Edit schedule** to control discovery. Pausing discovery leaves existing alerts monitoring; pause those individually if needed. See the [schedule API reference](/docs/api#scheduled-pair-discovery) for integration and deduplication details.

## Remove common exposure from relative-value signals

In **Pairs → Advanced strategy setup → Exposure-adjusted relative value**, choose a relationship family, target and reference basket. Specify distinct market, sector and factor ETF hedges. **Analyze residual** shows the adjusted score, estimated before/after betas and every proposed hedge leg. **Watch residual** saves a persistent monitor. **Add to scheduled search** builds a list of economic relationships; enable its schedule below the analysis form.

**Save C++ backtest strategy** puts the configured strategy in Research. Load that saved strategy, choose a daily date range with sufficient warmup, and run it with realistic costs. Six non-event families are also available under the C++ starter-strategy selector. The report compares performance and risk to SPY for the same evaluated period. See [relative-value models](/docs/api#relative-value-models) for the mathematics, event inputs and current instrument limits.

## Automate strategies on your stock lists

Open **Pairs**, name a list, and enter 2–20 stocks. Select the strategy families and choose **Save list & start automatic checks**. The first scan runs shortly; subsequent scans run on the saved interval while the backend is running. Lists, schedules, monitors and alert history survive restarts.

Residual pairs and baskets use broker industry/category profiles to identify related stocks, then remove fitted market, sector and factor ETF exposures. Daily lead-lag is optional. Missing classifications are reported; an explicit sector override treats the list as a relationship you supplied. Supported share-class and ETF checks cover GOOG/GOOGL, SPY/IVV/VOO and GLD/IAU when both instruments are in your list. Hedge ETFs may be outside the list.

Your action appears on each result:

- **Wait**, **Paused**, unavailable/stale data, or a score beyond the stop band: no buy/short instruction is shown.
- **Entry candidate**: review every **BUY / LONG** and **SELL SHORT** leg, including the exposure hedges. Weights are percentages of gross exposure, not share quantities. The app does not submit these trades.
- Open **Why this signal? Fundamentals & backtest** for the relationship, residual score, fitted hedges and reported company fundamentals with filing links. Fundamentals provide context and are not inputs to these price-based signals.
- **Backtest this setup** runs the configured C++ strategy over a year ending at the signal date and compares it with SPY. The displayed assumptions include capital and transaction costs. This is historical evaluation, not an open-position manager.

Editing the list queues another scan and pauses monitors it created whose target/reference stocks were removed. Deleting the list stops its schedules and pauses its generated monitors; history stays available. Existing reused monitors retain their settings. Changing discovery settings affects new monitors. Pausing a schedule does not pause its existing monitors. The monitor cap includes paused rules; remove unwanted monitors in Advanced strategy setup to free capacity.

## Screen fundamental valuations

In Pairs, edit a saved list and select **Peer fundamental valuation**. Enter additional reference peers if the tradable list is small. Each target needs at least 10 other eligible companies in the same SEC industry group. Reference-only symbols improve estimation without becoming trade candidates. Save the list to enable scheduled checks.

The **Peer-relative fundamental valuation** panel shows earnings multiples adjusted for revenue growth, profitability, cash generation and liabilities/assets. Open **Why this valuation?** to inspect the peers, contributions and dated source filings. Missing financials and insufficient peers are explained. This version supports profitable US-GAAP nonfinancial companies with one listed security; it does not apply the same formula to banks, funds or loss-making companies.

A valuation discount alone does not produce a buy instruction. Choose an explicit sector ETF in optional settings to evaluate a confirming price residual and exposure hedges. Only confirmed candidates show buy/short legs; scheduled entries are saved in persistent history. Historical fundamental-strategy backtesting is not enabled in this release. See the [model reference](/docs/api#peer-relative-fundamental-valuation) for the estimation method and limits.
