# StockAssist

A local stock research workspace with candlestick charts, saved one-shot price alerts, and a persistent paper trading account. The web MVP runs without cloud credentials.

## Run

Requirements: Node.js 22+, npm, CMake, a C++20 compiler, and the [native backend dependencies](backend/README.md). Python 3.11+ supplies optional Python strategy support and test tooling. From this directory on macOS/Linux:

```sh
npm run setup
npm run dev
```

Open http://127.0.0.1:3000. Backend health: http://127.0.0.1:8000/health. Build and architecture guide: [backend/README.md](backend/README.md). Stop both services with Ctrl+C. Ports 3000 and 8000 must be available.

```sh
npm test
npm run lint
npm run typecheck
npm run build
```

For a production-mode local frontend, run the backend separately and then `npm --prefix frontend run start -- --hostname 127.0.0.1` after building.

## What works

- Search by symbol or company, with eight sample symbols in demo mode.
- Responsive daily/intraday candlesticks down to one second, older-history paging, and individual trade snapshots.
- Create, edit/rearm, list, and delete persistent price alerts. A background worker checks every 30 seconds while the API runs; the dashboard refreshes alert status every 15 seconds.
- Buy and sell whole shares in a $100,000 paper account. SQLite transactions prevent overspending and overselling, including concurrent requests. Cash, positions, cost basis, and the last 50 fills persist across restarts.
- Optional Massive historical data provider, with timeouts, bounded short-lived caching, and explicit upstream errors.

## Data modes

The default `STOCKASSIST_DATA_MODE=demo` uses deterministic **synthetic** prices. These prices are not historical market observations. Demo weekdays are illustrative and do not model exchange holidays.

To use actual historical daily bars, set these in `backend/.env` (see `.env.example`):

```dotenv
STOCKASSIST_DATA_MODE=massive
MASSIVE_API_KEY=your_key
```

Existing `POLYGON_API_KEY` is accepted as a fallback key only when Massive mode is explicitly selected. The integration follows the [Massive daily aggregate API](https://massive.com/docs/rest/stocks/aggregates/custom-bars). Access depends on your provider plan. Failed provider requests return errors; they never silently substitute demo data. Provider mode requires validation with your account and is not a live quote feed.

Data is stored in `backend/data/demo.sqlite3`, `backend/data/ibkr.sqlite3`, or `backend/data/massive.sqlite3`, keeping provider accounts separate. Override `STOCKASSIST_DB` to choose another location. Back up the database while the API is stopped. No database reset happens on startup.

Paper execution uses live IBKR bid/ask quotes: buys pay the ask and sells receive the bid, with a modeled $0.005/share commission ($1 minimum). Market and limit orders are immediate-or-cancel: displayed size caps the fill and the remainder is cancelled. Delayed/frozen, disconnected, stale, or crossed quotes cannot fill. Live orders require regular US hours and a recent trade; short selling is not supported. This is local simulation, without broker routing, depth-based market impact, dividends, or corporate-action adjustments. Demo mode uses explicitly synthetic quotes. Alerts also evaluate daily closes, not intraday crossings. Triggered alerts remain visible until deleted or rearmed; email/push delivery is not implemented.

## Scope and architecture

- `frontend/`: Next.js dashboard and server-side API proxy. `STOCKASSIST_API_URL` defaults to `http://127.0.0.1:8000`.
- `backend/native/`: C++ HTTP API, SQLite persistence, native IBKR SDK connection, alerts, paper execution, and backtest jobs.
- `backend/simulation/`: native simulation loop, strategy ABI, starter strategies, and the optional Python callback adapter.
- `backend/tests/python_reference/`: previous Python implementation, retained as test/reference material and never launched by the app.
- `stockalert/`: original Flutter counter starter, preserved; not part of the web MVP.

This version is a **local, single-user application**. Bind it to loopback. The existing optional Supabase auth starter is preserved for future work, but it does not isolate or authenticate the local paper account. Public/multi-user hosting requires authenticated API access and per-user storage first. Brokerage execution, mobile functionality, and external notifications are outside this local app. The research workspace now provides modular Python/C++ backtesting.

## Browser verification

After building, install Chromium once with `cd frontend && npx playwright install chromium`, then run `npm run test:e2e` from `frontend/`. Tests start isolated backend/frontend servers on ports 8100/3100 and use a temporary database; your paper account is not modified.

## Real IBKR data

StockAssist can fetch real **completed daily and intraday bars** and search USD stock symbols through TWS or IB Gateway using the native IBKR C++ SDK. Historical requests return completed bars. Charts also subscribe to live quotes through the C++ backend and poll the latest quote once per second, updating sampled candles without resetting the viewport. Use **Follow live** to return to the newest bar. Live position marks and paper fills use quotes; alerts continue to evaluate completed daily closes. Live API market-data subscriptions are required in addition to historical-data access.

1. Start TWS or IB Gateway and log in to your IBKR account.
2. In TWS API settings, enable **ActiveX and Socket Clients**, allow the local connection, and leave **Read-Only API** enabled. The app makes no broker order requests.
3. Match the socket port in `backend/.env`:

   ```dotenv
   STOCKASSIST_DATA_MODE=ibkr
   IBKR_HOST=127.0.0.1
   IBKR_PORT=7497
   IBKR_CLIENT_ID=71
   ```

   Typical defaults are TWS paper `7497`, TWS live `7496`, IB Gateway paper `4002`, and IB Gateway live `4001`; verify your application's actual setting. Use a positive client ID not used by another API application.
4. Restart `npm run dev`. Open the dashboard and select a stock. The header reports IBKR connection status. `/api/market/status` shows connection diagnostics without account identifiers.

Your IBKR login needs API market-data permissions/subscriptions for the requested instruments. Data errors display the IBKR error code and do not fall back to simulated prices. The data connection is made on demand and retried on the next request after a disconnect. Successful requests are cached for 60 seconds; symbol requests are spaced at least 1.1 seconds apart. Use one backend worker.

IBKR-mode paper trades and alerts have their own database, `backend/data/ibkr.sqlite3`. They do not use your IBKR cash balance, positions, or order execution.

References: [IBKR API setup](https://interactivebrokers.github.io/tws-api/initial_setup.html), [IBKR historical bars](https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-bars/receiving-historical-bars), and [IBKR C++ SDK](https://interactivebrokers.github.io/).


## Find stocks and backtest custom algorithms

Open [Research & backtest](http://127.0.0.1:3000/research) from the dashboard. Use IBKR gainers, losers, or volume scans with price/volume filters, or search a symbol directly. Paste/import Python or C++ strategy code, set dates, warmup, initial cash, fees, and slippage, then run. You can save strategy versions, revisit completed runs, and export the full result with code and input data.

The Python/C++ examples use the same modular simulation engine. Signals execute at the next bar’s open; the report shows an equity curve against buy-and-hold, returns, drawdown, Sharpe, and individual fills. Execution supports daily and intraday single-stock long-only strategies and multi-stock long/short portfolios. Strategy programs run locally in separate processes and must be trusted.

See the [strategy authoring and engine guide](backend/simulation/README.md) for both language interfaces, execution assumptions, limits, and extension points. C++ requires a C++17 compiler; on macOS, install Xcode Command Line Tools if needed.

## Pair trading alerts

Open [Pair alerts](http://127.0.0.1:3000/pairs). Choose Stock A and Stock B, preview the metric, then create a rule. The form's example symbols do not create any alerts automatically.

- **Log-spread z-score**: spread = `ln(A close) − beta × ln(B close)`. Beta is a fixed positive weight entered by you (default 1), not an estimated share hedge. The current completed spread is scored against the mean and population standard deviation of the **preceding** N aligned sessions, excluding the observation being scored. Default lookback: 60; supported range: 20–250.
- **Price ratio**: `A close / B close`, with an at-or-above or at-or-below threshold. The log-spread weight does not affect ratio alerts.
- Z-score rules support signed above/below levels, divergence outside `±level`, or convergence inside `±level`. Boundaries are inclusive. For example, an outside level of 2 watches `abs(z) >= 2`; inside 0.5 watches `abs(z) <= 0.5`.

A one-shot alert pauses after its first trigger. Enable **Repeat after the condition clears** to keep watching: a later completed bar must clear the condition before another false-to-true transition triggers. Repeated polls of the same close do not notify again. Rearming or editing explicitly starts a new alert revision, so it may trigger on the same close again. Conditions already met when an alert is created/rearmed trigger at the next worker check; crossing from the other side first is not required.

The background worker checks about every 30 seconds while the backend is running. The page refreshes status/history every 10 seconds. There is no historical notification backfill after downtime. Alerts use completed daily closes, not intraday quotes. Both legs must share their latest date; missing dates are intersected, not forward-filled. Bars older than seven calendar days, mismatched latest dates, insufficient shared history, or a zero-variance spread do not trigger alerts and produce visible evaluation errors. A subsequent valid check clears the error.

Pair alerts and notifications persist in the current mode's SQLite database. Edit, pause, rearm, or delete rules from the page. Deleting a rule retains its earlier notifications. Notifications are local/in-app only; email, push delivery, and brokerage pair execution are not enabled. Pair-alert controls are separate from portfolio backtest execution and the local paper account.

Z-score alerts measure deviation only. They do not test cointegration, fit an optimal hedge ratio, or establish that a pair is suitable to trade. For background on statistical pair construction, see [QuantConnect's pairs research guide](https://www.quantconnect.com/docs/v2/research-environment/applying-research/pca-and-pairs-trading).

### Chart history and backtest performance

Pan left or zoom out on a stock candlestick chart to fetch older daily history automatically. Pages use the earliest loaded date as an exclusive cursor; adding bars preserves the visible window. Loading/errors and a retry control appear below the chart. Availability depends on the provider and symbol.

Backtests now use a C++ event loop for both Python and C++ strategies. Repeated identical runs use a persistent result cache; the lab identifies cache hits and offers Force rerun. Market snapshots expire after one hour or a new New York calendar date. See `backend/simulation/README.md` for invalidation and strategy contracts.

### Fine-grained IBKR data

Use **Candle interval** on the dashboard or **Backtest interval** in Research to select candles down to one second. Intraday pages load as you pan; the lab accepts UTC start/end times and runs the chosen interval through the C++ engine. Historical pages respect IBKR pacing and retention limits.

Expand **Individual trades · IBKR Time & Sales** for separate trade records and JSON export. This is a historical snapshot with second-resolution timestamps, not a live subscription. Multiple trades in a second remain separate.

### Company fundamentals and news

The stock dashboard includes SEC-reported company fundamentals with per-metric reporting periods and links to source filings, plus recent company-related headlines from your IBKR API news providers. **Read article** opens a text preview; **Refresh company research** checks again. Changing the selected stock updates both panels. Annual financial results are distinguished from newer balance-sheet dates and are not labeled as live or TTM estimates. Optional Massive credentials add company profiles and valuation ratios. Missing provider access is shown explicitly, and demo data is labeled synthetic. See `backend/README.md` for SEC ticker-mapping coverage and provider configuration.

### Portfolio backtesting and risk

Research now accepts up to ten stocks in one C++-simulated portfolio, with signed long/short weights and Python or C++ strategy code. Load the cointegration-pair or equal-weight-basket starter to get started. Every new run includes SPY buy-and-hold as an S&P 500 price-return reference on the same aligned bars, alongside stock/basket buy-and-hold. The result chart and risk table compare drawdown, volatility, historical tail losses, beta, and gross/net exposure. Default short-borrow costs and execution assumptions are editable/documented in the research workspace.

Pair discovery can filter on Engle–Granger cointegration evidence and find positive or inverse relationships. Previews show spread deviation, test statistics, fitted hedge, and half-life; signals identify both legs' directions. See `backend/simulation/README.md` for the portfolio callback contract, statistical definitions, and model limitations.
