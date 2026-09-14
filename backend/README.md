# StockAssist C++ backend

The running backend is `build/native/stockassist-server`, built from `native/src/`. It serves the existing API on `127.0.0.1:8000` and keeps the same per-provider SQLite databases under `data/`.

## Build and run

From the workspace root:

```sh
npm run setup
npm run dev
```

For backend-only development:

```sh
npm run build:backend
backend/build/native/stockassist-server --port 8000
```

Build requirements are CMake, a C++20 compiler, SQLite, libcurl, OpenSSL, Protobuf (including its matching `protoc`), and Python development headers for the optional Python adapter. Dependencies already installed in `/opt/homebrew` are detected on this Mac.

The IBKR adapter uses the existing licensed C++ SDK at the sibling `Quant/client` directory on this machine. That source tree is read only; generated Protobuf files go into this project's build directory. Elsewhere, set `IBKR_CPP_SDK_DIR` to an installed SDK client directory containing `EClientSocket.h`, `proto/`, and the SDK's decimal support. No IBKR SDK sources are redistributed in this repository. This build has been verified against the local SDK, including its `DecimalShim.cpp` compatibility layer.

`backend/.env` still selects `STOCKASSIST_DATA_MODE=ibkr|massive`, `STOCKASSIST_DB`, `IBKR_HOST`, `IBKR_PORT`, and `IBKR_CLIENT_ID`. Existing environment variables override the file. IBKR access remains read only; orders use the local paper ledger.

## Active modules

| File | Responsibility |
|---|---|
| `native/src/server.cpp` | HTTP routes, local-access checks, alert scheduling |
| `native/src/common.cpp` | Validation, UTC dates, configuration, SQLite, process limits |
| `native/src/ibkr.cpp` | Native SDK connection, contracts, history, ticks, scanner, pacing/cache |
| `native/src/market.cpp` | IBKR and Massive market data |
| `native/src/pairs.cpp` | Pair statistics, alert state/events, atomic paper trades |
| `native/src/backtests.cpp` | Bounded job queue, history paging, strategy compilation, persistent cache |
| `native/src/simulation.cpp` | Bar validation, statistics, and JSON reports |
| `simulation/native_engine.cpp` | Simulation loop, fills, fees, positions, benchmark and drawdown |
| `native/src/worker.cpp` | Separate native strategy process |
| `native/src/python_adapter.cpp` | Optional embedded-Python plugin |
| `simulation/python_strategy.py` | Python user-strategy context and module loading |

Neither the server nor the C++ worker links to Python. The Python plugin is loaded only when a run selects a Python strategy. C++ runs perform no Python execution. Python scripts remain for user strategies, tests, and the old implementation in `tests/python_reference/`; that reference app is never launched by `npm run dev`.

## Verification

```sh
npm test
npm --prefix frontend run build
npm --prefix frontend run test:e2e
```

`npm test` builds the native targets, runs C++ accounting/pair/concurrency checks, then tests the C++ HTTP API, persistence, caching, native/Python strategy parity, error containment, and intraday paging. Browser tests also launch the C++ server. Independent numerical comparisons use `tests/reference_engine.py`.

## Data and strategy semantics

The frontend's routes and saved SQLite tables are preserved. Old saved strategies and completed backtest reports remain readable. The native engine has a new cache fingerprint, so its first run warms a new cache rather than claiming an old engine's result.

Read `simulation/README.md` for the Python/C++ strategy contract, intervals, snapshot freshness, accounting assumptions, and process limits. Arbitrary user strategy code remains trusted local code, not a security sandbox.

Pair discovery: `POST /api/pairs/discover` accepts `tickers` (2–20 distinct USD stock symbols), `lookback` (20–250 trading days), `min_correlation` (0–1), and entry `threshold` (default 2). It ranks positive Pearson correlations of aligned daily log returns, fits the log-price hedge coefficient on the preceding window excluding the signal close, and returns preview snapshots and reusable alert configurations. Individual data failures are reported separately. Correlation is a screening measure, not a cointegration test. Z-score entry notifications identify the long and short leg; ratio and inside-band notifications do not imply an entry direction. Notifications are persisted in the app and checked while the server is running, using completed daily closes.


Live quotes: `GET /api/stocks/{ticker}/quote` maintains an IBKR streaming `reqMktData` subscription. A native reader pump dispatches callbacks; subscriptions expire after 60 seconds without requests, with at most 32 active symbols. Responses include bid/ask prices and displayed sizes, last-trade timestamp, receipt timestamps, and IBKR market-data type/errors. Chart updates sample quotes once per second; they do not reconstruct every tick or live traded volume. Historical bars remain the source for backtests. Massive mode has no live execution feed.

Paper execution: `POST /api/order` accepts `order_type: market|limit`, `limit_price` for limits, and an optional unique `request_id` for persistent idempotent retries. Whole-share long-only orders use IOC handling, bid/ask pricing rounded conservatively to cents, displayed-size partial fills, and $0.005/share fees ($1 minimum). Nonmarketable limits and invalid quotes reject without fills. Any unfilled quantity is cancelled, not queued. Quote validation runs again inside the SQLite transaction. Repeated fills cannot reuse the same quote's consumed displayed liquidity. US regular hours, a quote no older than 15 seconds, and a trade no older than 60 seconds are required in IBKR mode; missing entitlements, frozen/delayed feeds, and stale data block execution. The model does not simulate exchange queue priority, full depth, short borrowing, or IBKR's actual fee schedule. `order_details` adds fill metadata without changing historical order records.

Company research: `GET /api/stocks/{ticker}/fundamentals` reads SEC Company Facts in C++, selecting reported US-GAAP annual income/cash-flow figures and the latest point-in-time balance-sheet figures. Every available metric retains its reporting period, filing date, and filing link. Missing values remain null; quarterly, year-to-date, and annual durations are not mixed. SEC responses are cached for one hour, ticker mapping for one day, and failed SEC requests for five minutes; requests are serialized at no more than two per second. `SEC_USER_AGENT` optionally supplies your application/contact identification. The dashboard's default operating-company tickers have built-in CIK mappings; other symbols need access to SEC's ticker directory, which may be blocked by SEC. Funds and issuers without matching US-GAAP facts can have unavailable metrics. The current environment can retrieve Company Facts but returns HTTP 403 for the ticker directory.

`GET /api/stocks/{ticker}/news` uses subscribed IBKR API news providers in IBKR mode and otherwise tries Massive. Headlines include publisher codes and UTC timestamps. `GET /api/news/article?provider=...&article=...` retrieves IBKR article text plus optional provider HTML. The UI recreates an allowlist of paragraphs, headings, lists, tables, emphasis, and HTTP(S) links; provider scripts, styles, embedded resources, and event handlers are discarded. Plain-text articles retain their line breaks. PDF articles remain accessible in TWS. News requests use the broker's 60-second cache. Optional `MASSIVE_API_KEY` (or `POLYGON_API_KEY`) enriches company profiles/valuation ratios independently of IBKR price mode and provides a news fallback; key and subscription failures leave SEC metrics available and show a provider notice. This session's existing key returns HTTP 401, while real SEC data and IBKR headlines/article text were verified successfully.

Price alerts keep watching after a trigger. The C++ worker persists the last evaluated close and condition in `price_alert_state`, and records each crossing in `price_events` in the same SQLite transaction. `GET /api/alerts/events` returns the latest 100 saved notifications. `POST /api/alerts/{id}/state` with `{"active": false}` pauses a rule; `true` resumes it without clearing its crossing latch. Editing/rearming resets the latch. Deleting a rule retains events. New pair configurations default to `repeat: true`; explicit one-shot configurations remain supported.

The production server defaults to IBKR and rejects non-provider data modes. Test data lives in `tests/fixtures`, linked only into `stockassist-unit` and the explicitly built `stockassist-test-server`; it is not a runtime option in the application. Build the test server with `cmake --build backend/build/native --target stockassist-test-server -j 4` from the project root before running native HTTP tests.

## Client integration documentation

Start with the [HTTP API guide](../frontend/public/docs/api.md) for endpoint/request/response tables and the [working standard-library Python client](../examples/api/run_backtest.py) for submitting either C++ or Python strategy files. The [strategy guide](../frontend/public/docs/strategies.md) documents both callback contexts. All guides are also readable inside the running app at `/docs`.
