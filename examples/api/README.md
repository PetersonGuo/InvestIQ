# Run strategies from your own program

These clients submit a source file to the real StockAssist API, poll the job and save the report. They do not place paper or brokerage orders. Start the application with `npm run dev` first. Commands below run from the repository root.

## Python standard-library client

```sh
python3 examples/api/run_backtest.py --help
python3 examples/api/run_backtest.py \
  --source backend/simulation/examples/moving_average.cpp \
  --tickers AAPL --params '{"fast":10,"slow":30}' \
  --warmup 30 --output result.json
```

The default daily window covers the last 180 calendar days. Use `--start` and `--end` for explicit dates. Intraday runs require explicit UTC timestamps and `--interval`. Use `--force-rerun` when changing external strategy inputs. The client itself is Python, but a submitted `.cpp` algorithm runs entirely in the C++ backtester.

## C++ client

```sh
c++ -std=c++17 -O2 -pthread -I backend/native/vendor \
  examples/api/run_backtest.cpp -o /tmp/stockassist-client
/tmp/stockassist-client backend/simulation/examples/moving_average.cpp \
  AAPL 2026-06-01 2026-09-01 result.json
```

Choose available dates. Arguments are source, comma-separated tickers, start date, end date, output file and optional port. Its request uses daily bars, warmup 30, empty parameters, $100000 initial cash, $1 commission and 5 bps slippage; edit the JSON payload for other settings. Source can be `.cpp` or `.py`.

Both clients overwrite the named output file. Polling timeouts do not cancel the server job; retain the printed run id to resume with `GET /api/backtests/{id}`. Neither client automatically retries POST after a connection failure.

Read the [HTTP API](../../frontend/public/docs/api.md) and [strategy callback guide](../../frontend/public/docs/strategies.md), also available in the running app at `/docs`.
