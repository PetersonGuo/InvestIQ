# HTTP API reference

[Start here](start.md) for setup and a runnable client. [Strategy callbacks](strategies.md) describe the source code passed to this API.

## URLs, requests and errors

Your own Python/C++/terminal program should use http://127.0.0.1:8000. Health is GET /health; application routes start with /api. The web frontend uses its own proxy at http://127.0.0.1:3000/api. Through that proxy health is /api/health, not /health. Do not add two /api prefixes.

Use Content-Type: application/json for POST and PUT. There is no bearer token for this local single-user API. Local Host headers are required. Browser write requests only allow the configured local frontend origins; requests from arbitrary websites are rejected. A command-line client normally sends no Origin header. Keep provider keys and IBKR configuration on the backend, not in request bodies.

Successful responses contain JSON except DELETE, which returns an empty 204 response. Request bodies are capped at 128 KiB. Numeric fields must be JSON numbers, not numeric strings. Boolean fields must be JSON booleans. Error responses normally have this shape:

```json
{"detail":"Start date must be before end date."}
```

| HTTP status | Meaning and action |
| --- | --- |
| 200 | Read succeeded; inspect body for job failures, quote errors or provider notes |
| 201 | Strategy version, alert or paper order created |
| 202 | Backtest submission accepted; may already be completed on a cache hit |
| 204 | Delete succeeded; do not parse JSON |
| 400 / 403 | Invalid local host or disallowed browser origin |
| 404 | Unknown resource, symbol or unavailable history |
| 409 | Paper execution rejected or idempotency key conflicts |
| 413 | Request body exceeds server limit; error may not be JSON |
| 415 | Missing or incorrect Content-Type |
| 422 | Invalid fields, unsupported interval/mode, or strategy/data validation failure |
| 429 | Backtest concurrency, live-symbol cap or historical pacing budget reached; follow detail and retry later |
| 500 | Backend failure; inspect local server logs |
| 502 / 503 / 504 | Provider unavailable, disconnected, rejected request or timed out |

Do not blindly retry writes. Backtests and paper orders have different deduplication rules described below. There is no job-cancel endpoint and no public OpenAPI/Swagger endpoint currently.

## Health and provider status

```sh
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/api/market/status
```

Health reports status, backend, engine_version, data_mode, trading_mode and account_mode. Backend availability is independent of market-data availability. In IBKR mode market/status includes provider, connected, host, port, client_id, transport and last_error. In Massive mode connected is null because there is no persistent broker connection.

## Search, historical bars and live quotes

| Method and path | Inputs | Main response fields |
| --- | --- | --- |
| GET /api/search | ticker: symbol/company search text, up to 100 characters; limit: 1–320, default 20 | results with ticker/name; source |
| POST /api/search | Same fields as JSON | Same as GET |
| GET /api/stocks/{ticker} | interval: default 1d; days: 2–250, default 90; before: optional exclusive cursor | ticker, name, source, as_of, price, change_percent, interval, bars, has_more |
| GET /api/stocks/{ticker}/quote | No body; IBKR only | ticker, source, bid, ask, last, sizes, receipt/trade timestamps, connected, market_data_type; optional error |
| GET /api/stocks/{ticker}/ticks | before: optional UTC timestamp; IBKR only | ticker, source, ticks, timestamp_resolution, session, as_of |
| POST /api/scanner | scan_code, min_price, max_price, min_volume, limit | results with rank/ticker/name/exchange, source, scan_code, as_of |

```sh
curl --fail-with-body 'http://127.0.0.1:8000/api/search?ticker=Apple&limit=10'
curl --fail-with-body 'http://127.0.0.1:8000/api/stocks/AAPL?days=90&interval=1d'
curl --fail-with-body 'http://127.0.0.1:8000/api/stocks/AAPL/quote'
curl --fail-with-body -H 'Content-Type: application/json' \
  -d '{"scan_code":"HOT_BY_VOLUME","min_price":5,"max_price":1000,"min_volume":100000,"limit":20}' \
  http://127.0.0.1:8000/api/scanner
```

Bars are chronological objects containing time, open, high, low, close and volume. time is a daily date or an intraday UTC ISO timestamp. price is the final historical close, not a live quote. To fetch older data, send the oldest returned bars[0].time as before. Deduplicate by time when merging. Stop on an empty page or has_more:false; has_more:true is a paging hint, not a guarantee that older data exists. An empty older page may omit price/name/as_of.

The accepted intervals are 1s, 5s, 10s, 15s, 30s, 1m, 2m, 3m, 5m, 15m, 30m, 1h, 4h and 1d. Despite its name, days is the requested count of daily bars, not calendar days; intraday page size is determined by the IBKR duration for that interval. before uses YYYY-MM-DD for daily data and a UTC timestamp for intraday data. Intraday bars and tick records require IBKR. Massive supports daily bars.

Quote requests maintain a backend IBKR streaming subscription, but this HTTP API is polling, not WebSocket/SSE. Poll around once per second and inspect errors and timestamps; a last price alone is insufficient for execution. market_data_type 1 means live, 2 frozen, 3 delayed and 4 delayed-frozen. Subscriptions expire after 60 seconds without requests, with at most 32 active symbols. Missing subscriptions remain explicit errors. Tick records are historical snapshots, not a stream; keep multiple records with the same timestamp.

Scanner codes are HOT_BY_VOLUME, TOP_PERC_GAIN and TOP_PERC_LOSE. Defaults are shown in the request above. min_price accepts 0–100000, max_price is positive and at most 100000, min_volume is an integer 0–1000000000, and limit is an integer 1–50. min_price cannot exceed max_price. The scanner requires IBKR.

Small-bar and tick requests have a local budget of 55 requests per ten minutes, and identical small-bar requests must be at least 15 seconds apart when not served from cache. Direct requests can return HTTP 429; a backtest can instead fail with that pacing message in its error field. Follow the message, use fewer symbols/a smaller window or choose a coarser interval.

## Company fundamentals and news

| Method and path | Inputs | Response |
| --- | --- | --- |
| GET /api/stocks/{ticker}/fundamentals | Ticker in path | SEC metrics, reporting dates, filing links, optional Massive enrichment, notes |
| GET /api/stocks/{ticker}/news | Ticker in path | articles, source, fetched_at, notes |
| GET /api/news/article | provider and article query parameters from a headline | text, optional html, source |

A headline has id, title, publisher, published_at and url. IBKR headlines also have provider and article_id; they commonly have no public publisher URL. URL-encode provider/article_id and request /api/news/article?provider=...&article=.... The article page in the frontend is /news/article, while the JSON endpoint is /api/news/article.

Article html is untrusted provider markup. StockAssist's frontend recreates only allowed structures and HTTP(S) links. If you build your own UI, sanitize it before rendering; never insert it as unrestricted HTML. text is the compatibility fallback; HTML carries the original formatting. PDF articles are currently unsupported by the reader and return an error directing you to TWS.

Provider failures can produce HTTP 200 with empty articles/metrics and explanatory notes. Check those fields. News may fall back from IBKR to Massive, but never to invented articles. Fundamentals have per-metric periods; do not treat an annual metric as a live value.

## Submit and poll a backtest

POST /api/backtests always takes source code and settings. It does not accept a strategy_id in place of code. The program in examples/api/run_backtest.py handles the whole workflow; these are the underlying requests.

```sh
curl --fail-with-body -H 'Content-Type: application/json' \
  --data-binary @- http://127.0.0.1:8000/api/backtests <<'JSON'
{
  "name": "My first strategy",
  "language": "python",
  "code": "class Strategy:\n    def on_bar(self, ctx):\n        return 1.0 if ctx.portfolio.shares == 0 else None\n",
  "ticker": "AAPL",
  "start_date": "2026-06-01",
  "end_date": "2026-09-01",
  "interval": "1d",
  "params": {},
  "initial_cash": 100000,
  "commission": 1,
  "slippage_bps": 5,
  "warmup": 0,
  "force_rerun": false
}
JSON
```

Choose dates for which your provider has bars; the dates above illustrate a request, not a claim about data availability. A new job returns HTTP 202 and an object like {"id":"<run-id>","status":"queued","cache_hit":false}. Replace <run-id> below with the returned id:

```sh
RUN_ID='paste-the-returned-id-here'
curl --fail-with-body "http://127.0.0.1:8000/api/backtests/$RUN_ID"
```

Set RUN_ID to the returned id before running that command. Poll once per second until status is completed or failed. The state flow is queued → running → completed or failed. A cache hit may return status completed immediately. GET returns id, status, created_at, request, result and error. result is null until completion. A failed job still returns HTTP 200; read status and error. Validation can also reject the initial POST directly.

| Backtest field | Required/default | Contract |
| --- | --- | --- |
| language | Required | python or cpp |
| code | Required | Nonempty source string, up to 64000 UTF-8 bytes; one source file |
| name | My strategy | Nonempty, up to 100 UTF-8 bytes |
| ticker or tickers | Required | One ticker, or an ordered list of 1–32 distinct symbols; tickers takes precedence |
| start_date / end_date | Required | Start before end; daily dates or UTC intraday timestamps |
| interval | 1d | One of the intervals listed above |
| params | {} | At most 50 entries, finite numeric values only; no booleans, strings, arrays or nested objects |
| initial_cash | 100000 | Dollars; 100–100000000 |
| commission | 1 | Dollars per nonzero fill; 0–1000 and at most initial_cash |
| slippage_bps | 5 | Basis points applied adversely to execution; 0–1000; 5 bps = 0.05% |
| warmup | 30 | Integer 0–1000; at least warmup+2 shared bars must remain |
| max_gross_exposure | 1 | Portfolio target limit; 0.1–2.0; does not enable single-stock shorting/leverage |
| borrow_rate_percent | 3 | Annual percentage for portfolio short borrow; 0–100 |
| force_rerun | false | Skip result reuse and request a refreshed data snapshot |

Daily end_date is inclusive. Intraday start_date is inclusive and end_date exclusive; use timestamps such as 2026-09-01T13:30:00Z. Daily backtests select from the latest 500 available daily bars; they do not page arbitrarily far into the past. Intraday windows are limited to 31 calendar days, 40 pages and 50000 bars per stock. Portfolio data is capped at 200000 aligned asset-bars. Every strategy stock and SPY must share enough timestamps. The result reports actual coverage, which can differ from requested coverage.

## Read the result

| Path in the completed run | Meaning |
| --- | --- |
| request | Exact submitted source, parameters and normalized settings |
| result.data | Actual dates/count, primary bars, ordered tickers, all asset bars, SPY bars, pre-alignment counts, SHA-256 checksum |
| result.metrics | Strategy summary and reference returns; fields can be null when undefined |
| result.equity_curve | Chronological account equity, reference curves and drawdown data |
| result.fills | Executed historical trades with costs; differs from proposed signals |
| result.unfilled_final_signal | Last-bar instruction with no following open, or null |
| result.risk_comparison | strategy, spy and buy_and_hold reports on the same curve timestamps |
| result.risk_basis / assumptions | Frequency, measurement basis and execution limitations |
| result.logs | Up to 4000 characters of worker output |
| result.engine_version / engine_build | Runtime version and code fingerprint |

Each risk_comparison member contains these fields:

| Risk field | Units / interpretation |
| --- | --- |
| total_return_percent / annualized_return_percent | Percent; annualized value is null for intraday |
| max_drawdown_percent | Nonpositive percent from a prior equity peak |
| volatility_percent / downside_volatility_percent | Percent; annualized for daily runs, per-bar for intraday |
| sharpe_ratio / sortino_ratio | Ratios using a zero risk-free rate / zero downside target; null when undefined |
| var_95_percent / cvar_95_percent | Nonnegative historical 95% loss percentages for one observation |
| beta_to_spy | Ratio; null when SPY reference variance is insufficient |
| tracking_error_percent | Percent deviation from SPY returns at the stated frequency |
| average_gross_exposure_percent / max_gross_exposure_percent | Absolute position value divided by equity, in percent |
| average_net_exposure_percent | Signed position value divided by equity, in percent |
| time_in_market_percent | Percent of closing observations with invested exposure |

Inspect actual data coverage, costs, fills and exposure before interpreting a return. SPY is an S&P 500 price-return proxy. Risk statistics measure close-to-close observations, not intrabar extremes. Detailed fields and timing are explained in [the strategy guide](strategies.md).

GET /api/backtests lists the latest 30 run summaries. GET /api/backtests/{id} returns one complete report. There is no delete, update or cancel run route. Saving the GET response as JSON is the export operation; there is no separate /export endpoint.

## Cache behavior

Repeated identical requests can reuse a completed run, including across backend restarts. Identical pending submissions can share a job. Cache hits retain the original run id and name; renaming does not force execution. Only POST responses indicate cache_hit explicitly.

The key includes code, language, parameters, symbol order, dates, interval, costs and engine/build fingerprint. Data snapshots are reused for up to one hour or the next New York date. After a refresh, unchanged data can still reuse computation. Failed runs are not reused. force_rerun:true bypasses result reuse, but upstream provider pacing/cache limits still apply. External files, installed dependencies, randomness and network responses used by your code are not fingerprinted.

Two backtests can be active at once. On HTTP 429, wait for a known job to finish. A client-side polling timeout does not cancel the backend job; retain the id and resume GET polling later.

## Save and load strategy versions

GET /api/strategies/examples returns starter records with template_id, name, description, language, code, params, warmup and optional tickers. Pick a record matching the language and copy its code/settings into a backtest request; template_id alone is not executable input.

POST /api/strategies takes name, language, code, params and optional tickers using the same strategy-field validation. It returns a new id and saved strategy fields with HTTP 201. GET /api/strategies returns saved versions, including updated_at. Saving again creates a new record; there is no PUT/DELETE strategy route. Run settings such as dates, costs and warmup are not stored by this endpoint. To run a saved version, find it in the list, copy its code/language/params/tickers, and add the desired backtest settings.

## Price alerts

| Method and path | Body / behavior |
| --- | --- |
| GET /api/alerts | All persisted rules |
| POST /api/alerts | ticker, direction: above or below, threshold: positive number up to 10000000 |
| PUT /api/alerts/{id} | Full ticker/direction/threshold configuration; edits and rearms |
| POST /api/alerts/{id}/state | {"active":false} pauses; true resumes without clearing the crossing state |
| DELETE /api/alerts/{id} | Deletes the rule; retains past events |
| GET /api/alerts/events | Latest 100 persisted notifications |

Rules return id, ticker, direction, threshold, active (integer 0/1), triggered_at and created_at. Threshold comparisons are inclusive. The worker checks completed daily closes about every 30 seconds while the backend runs. An already-matched newly armed rule triggers on the next check. Repeating rules stay active; a later close must clear the condition before a new crossing triggers. Repeated polls/restarts do not duplicate an already-observed crossing. PUT explicitly resets that state.

Events include id, alert_id, ticker, direction, threshold, price, as_of and created_at. as_of identifies the market observation; created_at is notification time. These are local saved events, not email/push delivery or live intraday triggers.

## Pair discovery, previews and alerts

```sh
curl --fail-with-body -H 'Content-Type: application/json' \
  -d '{"tickers":["AAPL","MSFT","NVDA"],"lookback":60,"min_correlation":0.8,"relationship":"either","require_cointegration":true,"threshold":2}' \
  http://127.0.0.1:8000/api/pairs/discover
```

Discovery accepts 2–20 symbols, lookback 20–250 (default 60), min_correlation 0–1 (default 0.8), relationship positive/negative/either (default positive), require_cointegration boolean (default false), and threshold 0.1–20 (default 2). It returns pairs, errors, lookback and correlation_method. Each candidate contains correlation, config and snapshot. Partial data failures are listed in errors even when HTTP status is 200. Reuse a candidate's config to preview or create an alert; discovery does not create rules automatically.

| Pair configuration | Default and constraints |
| --- | --- |
| ticker_a / ticker_b | Required; distinct symbols |
| metric | zscore or ratio; default zscore |
| condition | above, below, inside or outside; default outside |
| threshold | Default 2; z-score -20 to 20; inside/outside must be nonnegative; ratio must be positive and at most 100000 |
| lookback | Integer 20–250; default 60 |
| hedge_ratio | Nonzero -100 to 100; default 1 |
| repeat | Boolean; default true |

Ratio rules only support above/below. Z-score baselines use the preceding lookback closes, excluding the latest signal observation. Correlation and cointegration are different fields; a configured z-score alert does not automatically stop when cointegration evidence weakens.

| Method and path | Behavior |
| --- | --- |
| POST /api/pairs/preview | Pair configuration → snapshot with value, z_score, ratio, matched, signal, cointegration, as_of and chart points; does not save |
| POST /api/pairs/alerts | Pair configuration → saved active rule |
| GET /api/pairs/alerts | Persisted rules with latest snapshot and last_error |
| PUT /api/pairs/alerts/{id} | Full pair configuration; edits and starts a new revision |
| POST /api/pairs/alerts/{id}/state | Required boolean active; pause/rearm starts a new revision |
| DELETE /api/pairs/alerts/{id} | Deletes rule, retaining event history |
| GET /api/pairs/events | Latest 50 saved notifications |

For trade direction, read signal.legs and signal.direction_text when signal is non-null; do not infer directions from correlation's sign alone. An inverse relationship can have both legs long or both short. Pair alerts use daily closes, retain history across restarts and do not backfill missed crossings while the backend was stopped. Rearming starts a new revision and may notify again on the same close. This differs from the price-alert state endpoint, which preserves its crossing state. There is no pair brokerage execution endpoint.

## Paper account and orders

GET /api/portfolio returns cash_cents, positions and the latest 50 orders. Money fields ending in _cents are integer cents; strategy cash/equity and backtest costs are in dollars. This is a local paper ledger, separate from your IBKR brokerage account and from each backtest's initial_cash.

POST /api/order takes the following JSON fields. Calling this endpoint changes the local paper account; neither a backtest nor a pair alert calls it automatically.

| Field | Meaning |
| --- | --- |
| ticker / side / quantity | Required symbol, buy or sell, and integer shares 1–3200000 |
| order_type | market (default) or limit |
| limit_price | Required for limit orders; dollars, 0.01–32000000000 |
| request_id | Optional string up to 100 characters; use a unique UUID for safe retries |

Orders are immediate-or-cancel: buys use the live ask and sells the bid, with displayed-size partial fills and the remainder cancelled. A non-crossing limit is rejected without a fill. Short selling is not supported in the paper account, even though portfolio backtests support shorts. Execution requires regular US trading hours, live non-delayed data, a quote no older than 15 seconds and a recent trade no older than 60 seconds. Paper fees are $0.005/share with a $1 minimum; backtest commission is a different configurable model.

A successful request returns HTTP 201 with its order record, including executed quantity, requested_quantity, cancelled_quantity, price_cents and fee_cents. For a retry, reuse request_id with the exact same JSON body; a successfully recorded order is returned instead of filled again. Reusing that id with different details returns 409. There is no cancel-order or broker-routing endpoint; unfilled IOC quantity does not remain open.

## Compatibility routes

GET / is a health alias. POST /api/alert/list and POST /api/alert/add are legacy aliases for listing and creating price alerts. New clients should use the plural /api/alerts routes above. The canonical contract is implemented in backend/native/src/server.cpp, with validation in common.cpp and the domain modules. The old Python reference backend under backend/tests is not the running API.

## Scheduled pair discovery

Schedules automatically discover pairs and create persistent repeating pair alerts. They run in the C++ backend, which must remain running; closing the browser does not stop them. These are in-app alerts, not broker orders or email notifications.

```bash
curl -X POST http://127.0.0.1:8000/api/pairs/schedules \
  -H 'Content-Type: application/json' \
  -d '{"source":"scanner","scan_code":"HOT_BY_VOLUME","limit":20,"interval_hours":24,"min_correlation":0.8,"require_cointegration":true,"relationship":"positive","lookback":60,"threshold":2,"max_alerts":5}'
```

The first run is queued immediately. Each successful run schedules the next run `interval_hours` after completion (elapsed time, not a market-close or wall-clock appointment). A failed run records `last_error` and retries after five minutes. After a backend restart, overdue schedules run once; missed intervals are not replayed.

| Method and route | Behavior |
| --- | --- |
| `GET /api/pairs/schedules` | List saved configuration, active/running state, next run, last start/finish, result and error. |
| `POST /api/pairs/schedules` | Validate and save an active schedule; returns 201 without waiting for IBKR. |
| `PUT /api/pairs/schedules/{id}` | Replace configuration, retain active state, and queue the next run immediately if active. Omitted settings use defaults. |
| `POST /api/pairs/schedules/{id}/state` | Body `{"active":false}` pauses; `true` resumes and queues a run. |
| `POST /api/pairs/schedules/{id}/run` | Queue an active, idle schedule now; returns 202, or 409 if paused/running. No body needed; the `Content-Type: application/json` header is still required. |
| `DELETE /api/pairs/schedules/{id}` | Delete the schedule; returns 204. Existing alerts and event history remain. |

| Configuration | Default and constraints |
| --- | --- |
| `source` | `scanner` or `symbols`. Scanner refreshes its universe from IBKR every run. |
| `tickers` | Required for `symbols`: 2–20 unique stock symbols. |
| `scan_code` | `HOT_BY_VOLUME`, `TOP_PERC_GAIN`, or `TOP_PERC_LOSE`; default `HOT_BY_VOLUME`. |
| `limit` | Scanner stocks: 2–20, default 20. |
| `min_price`, `max_price`, `min_volume` | Scanner filters: defaults 5, 1000, 100000. Minimum price must be below maximum. |
| `interval_hours` | Integer 1–168; default 24. |
| `max_alerts` | Integer 1–50; default 5. Caps this schedule's existing managed alerts, including paused alerts. |
| `lookback` | 20–250 completed trading days; default 60. |
| `min_correlation` | Absolute return correlation 0–1; default 0.8. |
| `relationship` | `positive` (default), `negative`, or `either`. |
| `require_cointegration` | Default true: require Engle–Granger evidence at 5%. |
| `threshold` | Entry z-score magnitude 0.1–20; default 2. |

For a saved universe, replace scanner fields with `"source":"symbols","tickers":["AAPL","MSFT","GOOGL","AMZN"]`. Discovery uses the same completed daily history, fitted hedge ratios, cointegration calculation and signal directions as `/api/pairs/discover`. It ranks candidates by absolute correlation and creates `zscore`/`outside` alerts with `repeat:true` up to the cap. A qualifying pair need not already be divergent: its alert waits for the threshold crossing. Inverse pairs can require both legs long or both short.

`last_result` contains `tickers`, `candidates` (count), `created` (pair labels), `managed_alerts` (count at that run), and per-symbol `errors`. No matches is a successful run if data loaded without errors. A run with no matches and data errors retries; a partial successful scan reports its errors alongside results.

The scheduler does not overwrite existing pair alerts, including manually created or paused rules. Deduplication treats A/B and B/A as the same pair. Existing rules retain their hedge ratios and thresholds when a schedule is edited or its universe changes. Deleting a managed alert frees capacity and suppresses that pair for that schedule, so it is not recreated next run. Lowering the cap does not delete older alerts. Pausing/editing/deleting during discovery invalidates the in-flight result. Pair monitoring continues independently, so pause individual alerts as well if you want their notifications to stop.

## Relative-value models

Open **Pairs → Exposure-adjusted relative value**. This uses the C++ return-residual model shared with the exported C++ strategy. It separates a configured economic relationship from selected market, sector and factor ETF return exposures. Neutrality is estimated, not guaranteed, and a residual is not proof of mispricing.

| Family | Implemented relationship | Scope |
| --- | --- | --- |
| `same_company` | Fixed dollar-return spread between US-listed share classes | Does not model ADR/local conversion, FX, foreign exchanges or fungibility. |
| `etf` | Fixed dollar-return spread between selected ETFs | Tracking relative value; not NAV or creation/redemption arbitrage. |
| `competitor` | Target versus competitor, with a jointly fitted reference and factor hedge | Relationships are supplied, not inferred from return correlation. |
| `basket` | Target versus 1–20 related stocks/ETFs, with jointly fitted hedge coefficients | Select economically meaningful constituents and enough estimation history. |
| `lead_lag` | Next-daily-bar target residual forecast from lagged reference residuals | Daily research only. No intraday/low-latency execution claim. Predictor assets are not automatically trade legs. |
| `event` | Supplied cash-plus-stock consideration, plus exposure-adjusted divergence | Requires terms, provenance and validity dates. Signal must agree with contractual payoff gap; deal failure risk remains. No event-news classifier. |
| `cross_asset` | US-listed ETF proxy versus stock/other ETF | Commodity ETF roll/carry remains. Direct futures calendars, bonds and convertibles are not supported contract models. |

### Analyze and monitor

```bash
curl -X POST http://127.0.0.1:8000/api/pairs/preview \
  -H 'Content-Type: application/json' \
  -d '{"relative_value":{"category":"competitor","target":"V","references":[{"ticker":"MA","weight":1}],"market":"SPY","sector":"XLF","factors":["MTUM","VLUE"],"fit_reference":true,"lookback":120,"entry_z":2}}'
```

The example is a configurable research starting point, not a claim of an opportunity. Post the same body to `POST /api/pairs/alerts` to save a repeating monitor. Existing alert state/delete routes work unchanged. `PUT /api/pairs/alerts/{id}` replaces a relative-value configuration when `relative_value` is present. Configuration persists in `pair_alerts`; notifications persist in `pair_events`. Alert checks refit from completed daily bars, normally on the existing 30-second worker loop; they are not tick-level alerts. The frontend shows these monitors and events in their own relative-value section.

| Configuration field | Meaning |
| --- | --- |
| `category` | One of the seven IDs above. |
| `target` | US-listed stock or ETF symbol. |
| `references` | 1–20 `{ticker, weight}` objects. `weight` is a fixed dollar-return coefficient when not fitting, not a share-conversion ratio. Fitted models ignore these fixed weights. |
| `market` | Market ETF proxy, default SPY. Must differ from target, references and other hedges. For a SPY/IVV relationship choose another market proxy explicitly. |
| `sector` | Required sector or broad-exposure ETF proxy. No automatic sector classification. |
| `factors` | Required 1–6 distinct additional ETF proxy symbols. For example a momentum and value ETF; these are not pure academic long/short factors. |
| `fit_reference` | Default true for competitor/basket; false otherwise. Jointly estimates reference coefficients when true. Lead-lag uses its own predictor fit; event always uses contractual share consideration. |
| `lookback` | 40–250 completed trading days, default 120. At least five observations per fitted coefficient; at least lookback + 3 aligned closes. |
| `entry_z` | Entry magnitude 0.1–20, default 2. For lead-lag this is predicted residual return divided by forecast residual noise, not a spread z-score. |
| `interval` | `1d` only. Other intervals are rejected. |
| `instrument_scope` | `us_stocks_etfs` only. Other scopes are rejected rather than resolved as US stocks. |
| `cash_per_share`, `shares_per_share` | Event consideration per target share, nonnegative and not both zero. Assumes same-currency USD securities. |
| `terms_as_of`, `expires_on`, `terms_source` | Required for event: YYYY-MM-DD dates and a nonempty provenance reference. Terms must be known by the observation date and remain valid. Backtests never apply the terms before `terms_as_of`. |

All symbols must be distinct. Histories must share a provider and latest completed date. Unavailable history, insufficient observations, collinearity, vanishing residual variance and unstable hedges fail explicitly. There is no synthetic fallback. Corporate actions, ETF distributions and price adjustment differences can contaminate return signals and must be checked with the provider data used.

### What is removed and what remains

For fixed spreads, define daily simple return `y[t] = r_target[t] - sum(w[j] * r_reference[j,t])`. Fit `y = intercept + market_beta * market_return + sector_beta * sector_return + sum(factor_beta * factor_return) + residual` on prior returns, excluding the latest signal return. Fitted competitor/basket models jointly regress target returns on reference and factor returns; this avoids sequentially removing market and sector exposures in an order-dependent way.

The implementation uses centered, standardized QR with reorthogonalization. It rejects rank-deficient exposures instead of using a penalty that silently leaves factor loading. It accumulates the prior fitted innovations into a residual path and compares the latest innovation plus that path against its prior mean/standard deviation. The intercept (estimated constant drift) is also removed. What remains is the unexplained return component; economic convergence and stationarity are not established by this regression. No two-variable cointegration p-value is reported for this multivariate model.

The suggested portfolio includes target, reference and all market/sector/factor hedge instruments. Raw weights `[1, -reference_coefficients, -factor_coefficients]` are normalized to 100% gross exposure and signed for the entry direction. Merely displaying a residual and trading only the target/reference legs would leave the factor risk in place.

Lead-lag first residualizes both target and leader returns against contemporaneous factors, fits target residual[t] from leader residual[t−1] using prior data, and forecasts the next residual. Its portfolio holds the target plus target-factor hedges; a predictive leader coefficient is not a contemporaneous hedge weight. A daily-bar backtest trades at the next open and therefore misses any forecast move occurring before that fill; it is not a simulation of immediate close-to-close execution.

The response adds `relative_value` with:

- `score_type`, `fit_r2` (in-sample), and `reference_coefficients`.
- `exposures`: proxy ticker, role, and estimated before/after beta, on the same gross-dollar scale and signal direction. In-sample after betas should be approximately zero.
- `hedge_legs`: ticker, long/short side, role and signed equity weight. Below the threshold these remain analytical candidate weights; `signal` is null.
- `event_gap_percent`: `100 * (target_price / (cash + shares * reference_price) - 1)` for supplied event terms, null otherwise.

Hedge weights refit over time. Future betas can drift, and actual whole-share holdings differ from ideal continuous weights. Dividend, financing, borrow availability and execution limitations remain those of the portfolio simulator.

### Schedule economic relationships

Use `source:"relationships"` in the existing schedule API. Supply 1–20 objects in `relationships`, each containing the complete `relative_value` configuration from the preview request above:

```json
{
  "source": "relationships",
  "interval_hours": 24,
  "max_alerts": 5,
  "relationships": [
    {
      "relative_value": {
        "category": "competitor",
        "target": "V",
        "references": [{"ticker":"MA","weight":1}],
        "market": "SPY",
        "sector": "XLF",
        "factors": ["MTUM","VLUE"],
        "fit_reference": true,
        "lookback": 120,
        "entry_z": 2
      }
    }
  ]
}
```

Each run evaluates configured relationships in list order, skips failed models with explicit errors, and creates repeating monitors up to the cap. Valid models need not already meet the entry threshold: monitors wait for divergence. Standard scanner `min_correlation` and `require_cointegration` filters do not apply to this source. A changed threshold/lookback does not duplicate or overwrite an existing monitor; edit the monitor itself to change it. Identity includes category, target, references, factor proxies, fixed weights and event terms. Raw two-stock rules and factor-adjusted rules are different models and can coexist. All existing pause/restart/retry/delete-suppression behavior applies.

### C++ backtesting

`GET /api/pairs/relative-value/catalog` returns seven research configurations and their limits. Six non-event families are also available under **Research → C++ → Starter strategy**. Event terms must be supplied before creating that strategy.

`POST /api/pairs/relative-value/strategy` with the same relative-value body returns editable `name`, `language:"cpp"`, `code`, `params`, `warmup`, and ordered `tickers`. It does not execute a trade or backtest. Save it with `POST /api/strategies`, or merge it with normal backtest dates/costs and submit to `POST /api/backtests`.

The callback includes `relative_value_math.h`, which the native compiler supplies with `strategy_api.h`. Its asset order is target, references, market, sector, then factors. It uses the same estimation code as preview, refits and rebalances all legs at each eligible next open, exits on the exit band, signal reversal, stop band, expired terms, or invalid model. Parameters include `reference_count`, `mode` (0 fixed spread, 1 fitted basket, 2 lead-lag), `reference_weight_N`, `lookback`, `entry_z`, `exit_z`, and `stop_z`. Event exports also include consideration and UTC validity timestamps. These templates reject intraday data. Up to 32 portfolio assets are supported to accommodate a 20-name reference basket plus factor hedges; the aggregate 200,000 asset-bar bound still applies.

The ordinary result cache, SPY buy-and-hold comparison over the same evaluated period, and portfolio risk report remain in force. Costs apply to every hedge-leg fill; borrow charges apply to all shorts. Invalid-model periods are flat or close existing holdings, so inspect fills and time invested as well as headline returns. Historical event terms are supplied by you, not reconstructed point-in-time by the app.

The ETF-factor residual approach is informed by [Avellaneda and Lee, Statistical Arbitrage in the U.S. Equities Market](https://math.nyu.edu/~avellane/AvellanedaLeeStatArb20090616.pdf). This implementation is a configurable research model, not a reproduction of their reported results.

## Saved stock lists and action results

These routes use the same local JSON API and error format as other pair routes.

| Method | Path | Request / result |
| --- | --- | --- |
| GET | /api/pairs/stock-lists | Saved lists with id, name, tickers, version and timestamps |
| POST | /api/pairs/stock-lists | Body: name (1–100 characters), tickers (2–20 distinct symbols); returns saved list |
| PUT | /api/pairs/stock-lists/{id} | Same body; replaces list and invalidates in-flight scans of the old version |
| DELETE | /api/pairs/stock-lists/{id} | Stops associated schedules and pauses monitors they created; retains history |
| GET | /api/pairs/opportunities | Compact relative-value monitors with discovery evidence and an action object |

Create the list first, then pass its id to POST /api/pairs/schedules:

```json
{
  "source": "stock_list",
  "stock_list_id": "ID_FROM_CREATE_RESPONSE",
  "algorithms": ["residual_pair", "basket", "same_company", "etf"],
  "interval_hours": 24,
  "max_alerts": 10,
  "lookback": 120,
  "threshold": 2,
  "min_correlation": 0.6,
  "market": "SPY",
  "sector": "auto",
  "factors": ["MTUM", "VLUE"],
  "basket_size": 5,
  "model_limit": 20
}
```

The optional fifth algorithm is lead_lag, using completed daily bars. Lookback is 40–250 observations, basket_size 2–10, model_limit 1–30 and factors 1–6 distinct symbols. Sector auto groups stocks by matching nonempty broker category and a recognized industry-to-sector ETF mapping. An explicit sector ticker overrides this eligibility check and represents a user-declared relationship. Market, sector and factors must be distinct. The correlation filter screens residual/basket/lead-lag candidates; supported same-company/ETF relationships have a separate economic mapping. Algorithms receive evaluation slots in turn up to model_limit.

The schedule last_result includes list/version, evaluated/eligible counts, created monitors, errors and skipped relationships. Missing provider data is reported, not replaced. Global deduplication can reuse an existing monitor without changing its parameters. The maximum managed-monitor count includes paused monitors. Removing a symbol pauses owned monitors that reference it; hedge symbols are exempt. Manual or reused monitors are not owned by the list and are not paused on deletion.

Each opportunity has action.state, action.label, action.legs, entry_z, exit_z and stop_z. States are entry, wait, paused, error, waiting_data, stale or avoid. Only entry exposes executable-direction legs; every other state returns an empty legs array. Legs include target/reference and market/sector/factor hedges. The feed suppresses entry instructions for data older than five calendar days or scores at/beyond the stop band. It does not check live borrow, spreads, capital or existing positions and never places an order. The exit/stop values describe the exported backtest strategy, not a managed live exit instruction.

Fundamentals are fetched separately through /api/stocks/{ticker}/fundamentals when the user opens the details. They are contextual reported data, not a valuation input or historical point-in-time guarantee. Use the existing relative-value export and backtest endpoints to reproduce the one-click backtest with custom costs and dates.

## Peer-relative fundamental valuation

POST /api/pairs/valuation screens current annual earnings valuations in C++. Supply tickers (2–20 tradable stocks) or stock_list_id. Optional valuation_peers supplies up to 40 additional estimation-only symbols. They cannot become targets or price-confirmation basket references unless also in the tradable list.

```json
{
  "tickers": ["QCOM", "AMD", "NVDA"],
  "valuation_peers": ["AVGO", "TXN", "ADI", "MCHP", "MPWR", "SWKS", "QRVO", "ON", "INTC", "MU"],
  "valuation_min_gap": 0.20,
  "sector": "XLK",
  "market": "SPY",
  "factors": ["MTUM", "VLUE"],
  "lookback": 120,
  "threshold": 2
}
```

This is an illustrative research universe, not a promise that every company qualifies or a trade recommendation. Add peer_valuation to a stock-list schedule's algorithms array to run it automatically. The schedule accepts valuation_peers and valuation_min_gap and stores the complete result under last_result.valuation. Failed price-pair screens do not erase a completed valuation result. List edits invalidate pending results; pausing a schedule suppresses its valuation actions in the UI.

The peer_earnings_v1 model predicts log annual P/E from revenue growth, net income/revenue, operating cash flow/revenue and liabilities/assets. It uses standardized ridge regression with fixed penalty 1 and an unpenalized intercept. Each target is excluded from its own training and error estimation. Peers require the same first three SEC SIC digits, at least 10 other distinct eligible issuers, positive annual earnings/revenue/assets, aligned annual periods, financials no older than 550 days, and P/E between 2 and 150. Financials, insurers, REITs, multiple listed share classes, unavailable US-GAAP facts and extreme ratios are excluded with reasons. Current SIC grouping needs business-model review; it does not establish identical exposure.

SEC company facts retain filing dates and accession links. Only facts filed strictly before the analysis date are selected; same-day filings are deferred. Current annual balance-sheet and income-statement periods must align. Prior revenue uses a consecutive annual period. Future restatements cannot overwrite a value before their filing date. Missing values are never replaced with zeros or synthetic financials.

Each row includes actual/expected multiples, a discount or premium (actual/expected minus one), peer symbols, additive log-multiple contributions, source evidence and action. The empirical valuation range uses the 90th-percentile held-out absolute log error, with finite-sample rounding and a minimum radius of 0.15. It is not a calibrated confidence interval. Features more than three training standard deviations from the mean produce extrapolation status and cannot generate entries.

Entry candidates require an estimate outside the empirical range, an absolute gap of at least valuation_min_gap (default 0.20), and a fresh confirming price residual in the same direction, beyond threshold but within the stop band. Price confirmation uses an equal-weight basket of up to five eligible tradable peers with market, explicit sector and supplied factor hedges. It needs at least two references. Sector auto permits valuation research but asks for an explicit sector ETF before price confirmation. All hedge legs are shown as fractions of gross exposure. No order is placed. Non-entry action.legs is empty.

GET /api/pairs/valuation/events returns the latest 100 persistent scheduled entry snapshots. Events are deduplicated by schedule, target, analysis date and direction. They remain after a list is deleted and are historical evidence, not current instructions. Normal schedule responses preserve snapshots even when paused; API clients must check active, last_error and data dates before treating one as current.

Historical as_of values are rejected. This release is current screening only: a fundamental-strategy backtest needs archived corporate-action-consistent prices/EPS, original filing vintages and historical universe membership. Existing price-residual backtests do not backtest this fundamental entry filter. Current screening still requires review of splits since the annual report, unusual earnings, peer comparability, borrow and execution costs. Market/sector/factor neutrality is an in-sample estimate, not a future guarantee.

## Cost-aware backtest results

New C++ relative-value exports include cost-aware entry and rebalance parameters, documented in the [strategy guide](/docs/strategies#cost-aware-relative-value-c-strategies). Portfolio results include metrics.total_slippage_cost in account currency, in addition to total_commission and total_borrow_cost. Optional strategy_diagnostics contains strategy-supplied counters explaining skipped entries/rebalances. Changing strategy source or parameters produces a distinct cache entry; earlier saved runs retain their original results.

To compare changes on exactly the same historical data, include replay_run_id with a completed run's ID in POST /api/backtests. Keep tickers (including order), interval, start_date and end_date identical to that run. Strategy code, parameters, warmup, capital and costs may change. The backend verifies the stored asset/benchmark data hash and bypasses provider requests. It preserves source metadata and returns data.replayed_from_run_id. Missing, incomplete, mismatched or corrupted snapshots fail explicitly. Force rerun still means recompute on those saved bars when replay_run_id is set; it does not fetch fresh bars.

In Research, load a completed run into the editor, then enable **Reuse loaded run’s exact historical data**. Edit the strategy parameters or costs and run it. This is an explicit dataset replay, distinct from reusing a cached strategy result.
