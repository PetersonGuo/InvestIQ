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

`backend/.env` still selects `STOCKASSIST_DATA_MODE=demo|ibkr|massive`, `STOCKASSIST_DB`, `IBKR_HOST`, `IBKR_PORT`, and `IBKR_CLIENT_ID`. Existing environment variables override the file. IBKR access remains read only; orders use the local paper ledger.

## Active modules

| File | Responsibility |
|---|---|
| `native/src/server.cpp` | HTTP routes, local-access checks, alert scheduling |
| `native/src/common.cpp` | Validation, UTC dates, configuration, SQLite, process limits |
| `native/src/ibkr.cpp` | Native SDK connection, contracts, history, ticks, scanner, pacing/cache |
| `native/src/market.cpp` | Market provider selection, demo and Massive data |
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
