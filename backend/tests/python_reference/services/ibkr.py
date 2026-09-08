"""Read-only IBKR data adapter. All socket operations run on one event loop.

No order/account synchronization or execution is requested. The application
continues to use its own paper ledger when this market-data provider is selected.
"""

import asyncio
import concurrent.futures
import math
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from collections import deque
from services.resolutions import RESOLUTIONS, utc_time
from fastapi import HTTPException
from ib_async import IB, Stock, ScannerSubscription
from ib_async.wrapper import RequestError


class IBKRProvider:
    def __init__(self):
        self.host = os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(os.getenv("IBKR_PORT", "7497"))
        self.client_id = int(os.getenv("IBKR_CLIENT_ID", "71"))
        if not 1 <= self.port <= 65535 or self.client_id < 1:
            raise ValueError(
                "IBKR_PORT must be 1–65535 and IBKR_CLIENT_ID must be positive."
            )
        self._start_lock = threading.Lock()
        self._loop = None
        self._thread = None
        self._ib = None
        self._cache = {}
        self._last_search = 0.0
        self._history_times = deque()
        self._identical_times = {}
        self._last_small_request = 0.0
        self._last_connection_failure = 0.0
        self._last_error = None

    def _start(self):
        with self._start_lock:
            if self._thread and self._thread.is_alive():
                return
            ready = threading.Event()

            def run():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._ib = IB()
                self._ib.RaiseRequestErrors = True
                self._request_lock = asyncio.Lock()
                ready.set()
                self._loop.run_forever()
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
                self._loop.close()

            self._thread = threading.Thread(
                target=run, name="stockassist-ibkr", daemon=True
            )
            self._thread.start()
            if not ready.wait(3):
                raise HTTPException(503, "IBKR client could not start.")

    def _connection_message(self):
        return (
            f"IBKR is disconnected at {self.host}:{self.port}. Log in to TWS or IB Gateway, "
            "enable socket API access, and match IBKR_PORT to its API port. "
            "Keep Read-Only API enabled. Then retry."
        )

    async def _connect(self):
        if self._ib.isConnected():
            return
        if time.monotonic() - self._last_connection_failure < 5:
            raise HTTPException(503, self._connection_message())
        try:
            # Connect the market-data transport only. IB.connectAsync also fetches
            # positions/account state, which this application does not need.
            await self._ib.client.connectAsync(
                self.host, self.port, self.client_id, timeout=4
            )
            self._cache.clear()
            self._last_error = None
        except (OSError, asyncio.TimeoutError, ConnectionError) as exc:
            self._ib.disconnect()
            self._last_connection_failure = time.monotonic()
            raise HTTPException(503, self._connection_message()) from exc

    async def _pace_history(self, key, small):
        if not small:
            return
        now = time.monotonic()
        while self._history_times and now - self._history_times[0] >= 600:
            self._history_times.popleft()
        if len(self._history_times) >= 55:
            wait = max(1, int(601 - (now - self._history_times[0])))
            raise HTTPException(
                429,
                f"IBKR historical request budget reached. Retry in {wait} seconds.",
                headers={"Retry-After": str(wait)},
            )
        if small:
            delay = max(
                0,
                0.45 - (now - self._last_small_request),
                15.1 - (now - self._identical_times.get(key, -1000)),
            )
            if delay > 3:
                raise HTTPException(
                    429,
                    "IBKR requires 15 seconds between identical small-bar requests. Retry shortly.",
                )
            await asyncio.sleep(delay)
            self._last_small_request = time.monotonic()
        self._identical_times = {
            k: v for k, v in self._identical_times.items() if now - v < 16
        }
        self._identical_times[key] = time.monotonic()
        self._history_times.append(time.monotonic())

    async def _execute(self, operation, value):
        async with self._request_lock:
            await self._connect()
            key = (operation, value, datetime.now(ZoneInfo("America/New_York")).date())
            cached = self._cache.get(key)
            if cached and time.monotonic() - cached[0] < 60:
                return cached[1]
            try:
                if operation == "search":
                    # IBKR symbol matching accepts at most one request per second.
                    await asyncio.sleep(
                        max(0, 1.1 - (time.monotonic() - self._last_search))
                    )
                    self._last_search = time.monotonic()
                    descriptions = await self._ib.reqMatchingSymbolsAsync(value)
                    if descriptions is None:
                        raise HTTPException(
                            504, "IBKR symbol search timed out. Retry shortly."
                        )
                    result = []
                    seen = set()
                    for description in descriptions:
                        contract = description.contract
                        if (
                            contract.secType == "STK"
                            and contract.currency == "USD"
                            and contract.symbol not in seen
                        ):
                            seen.add(contract.symbol)
                            result.append(
                                {
                                    "ticker": contract.symbol,
                                    "name": getattr(contract, "description", "")
                                    or contract.symbol,
                                }
                            )
                elif operation == "scanner":
                    filters = dict(value)
                    subscription = ScannerSubscription(
                        instrument="STK",
                        locationCode="STK.US.MAJOR",
                        scanCode=filters["scan_code"],
                        numberOfRows=filters["limit"],
                        abovePrice=filters["min_price"],
                        belowPrice=filters["max_price"],
                        aboveVolume=filters["min_volume"],
                    )
                    data = self._ib.reqScannerSubscription(subscription)
                    try:
                        future = self._ib.wrapper.startReq(data.reqId, container=data)
                        await asyncio.wait_for(future, timeout=7)
                        result = [
                            {
                                "rank": row.rank + 1,
                                "ticker": row.contractDetails.contract.symbol,
                                "name": row.contractDetails.longName
                                or row.contractDetails.contract.symbol,
                                "exchange": row.contractDetails.contract.primaryExchange,
                            }
                            for row in data
                        ]
                    finally:
                        self._ib.cancelScannerSubscription(data)
                elif operation in ("intraday", "ticks"):
                    symbol, interval, before = value
                    contracts = await self._ib.qualifyContractsAsync(
                        Stock(symbol, "SMART", "USD")
                    )
                    if not contracts or not contracts[0]:
                        raise HTTPException(
                            404, "IBKR could not identify this USD stock."
                        )
                    end = (
                        utc_time(before).strftime("%Y%m%d %H:%M:%S UTC")
                        if before
                        else ""
                    )
                    await self._pace_history(
                        key, operation == "ticks" or RESOLUTIONS[interval][2] <= 30
                    )
                    if operation == "ticks":
                        ticks = await self._ib.reqHistoricalTicksAsync(
                            contracts[0],
                            startDateTime="",
                            endDateTime=end
                            or datetime.now(timezone.utc).strftime(
                                "%Y%m%d %H:%M:%S UTC"
                            ),
                            numberOfTicks=1000,
                            whatToShow="TRADES",
                            useRth=True,
                            ignoreSize=False,
                        )
                        result = normalize_ticks(ticks)
                    else:
                        size, duration, seconds = RESOLUTIONS[interval]
                        if (
                            before
                            and seconds <= 30
                            and utc_time(before)
                            < datetime.now(timezone.utc) - timedelta(days=180)
                        ):
                            raise HTTPException(
                                422,
                                "IBKR small bars are available only for approximately the last six months. Choose 1 minute or larger for older history.",
                            )
                        bars = await self._ib.reqHistoricalDataAsync(
                            contracts[0],
                            endDateTime=end,
                            durationStr=duration,
                            barSizeSetting=size,
                            whatToShow="TRADES",
                            useRTH=True,
                            formatDate=2,
                            keepUpToDate=False,
                            timeout=7,
                        )
                        result = normalize_intraday(bars, seconds)
                        if before:
                            result = [
                                b
                                for b in result
                                if utc_time(b["time"]) < utc_time(before)
                            ]
                else:
                    await self._pace_history(key, False)
                    contracts = await self._ib.qualifyContractsAsync(
                        Stock(
                            value[0] if operation == "history_page" else value,
                            "SMART",
                            "USD",
                        )
                    )
                    if not contracts or not contracts[0]:
                        raise HTTPException(
                            404,
                            "IBKR could not uniquely identify this USD stock symbol.",
                        )
                    bars = await self._ib.reqHistoricalDataAsync(
                        contracts[0],
                        endDateTime=(
                            (value[1].replace("-", "") + " 00:00:00 UTC")
                            if operation == "history_page"
                            else ""
                        ),
                        durationStr="2 Y",
                        barSizeSetting="1 day",
                        whatToShow="TRADES",
                        useRTH=True,
                        formatDate=1,
                        keepUpToDate=False,
                        timeout=7,
                    )
                    result = normalize_bars(bars)
                    if operation == "history_page":
                        result = [bar for bar in result if bar["time"] < value[1]]
                    if not result and operation != "history_page":
                        raise HTTPException(
                            404,
                            "IBKR returned no completed daily bars. Check the symbol and historical market-data permissions in TWS.",
                        )
            except RequestError as exc:
                code = exc.code
                message = (
                    "IBKR rejected the data request "
                    f"(code {code}). Check market-data subscriptions/API permissions, "
                    "the stock symbol, and request pacing in TWS."
                )
                raise HTTPException(502, message) from exc
            if len(self._cache) >= 128:
                self._cache.clear()
            self._cache[key] = (time.monotonic(), result)
            self._last_error = None
            return result

    def request(self, operation, value):
        self._start()

        async def bounded():
            try:
                return await asyncio.wait_for(
                    self._execute(operation, value), timeout=12
                )
            except asyncio.TimeoutError:
                self._ib.disconnect()
                raise

        future = asyncio.run_coroutine_threadsafe(bounded(), self._loop)
        try:
            return future.result(timeout=13)
        except (concurrent.futures.TimeoutError, TimeoutError) as exc:
            future.cancel()
            self._last_error = (
                "IBKR request timed out. Check the connection and retry shortly."
            )
            raise HTTPException(504, self._last_error) from exc
        except HTTPException as exc:
            self._last_error = exc.detail
            raise
        except (ConnectionError, OSError) as exc:
            self._last_error = self._connection_message()
            raise HTTPException(503, self._last_error) from exc

    def status(self):
        return {
            "provider": "ibkr",
            "connected": bool(self._ib and self._ib.isConnected()),
            "host": self.host,
            "port": self.port,
            "client_id": self.client_id,
            "last_error": self._last_error,
            "price_type": "completed_daily_close",
        }

    def close(self):
        with self._start_lock:
            if self._loop and self._thread and self._thread.is_alive():
                self._loop.call_soon_threadsafe(self._ib.disconnect)
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=3)
            self._thread = None
            self._ib = None
            self._cache.clear()


def normalize_bars(bars, today=None):
    """Exclude the current US session: paper fills/alerts use completed closes."""
    today = today or datetime.now(ZoneInfo("America/New_York")).date()
    result = {}
    for bar in bars:
        day = bar.date
        if isinstance(day, datetime):
            day = day.date()
        elif isinstance(day, str):
            day = (
                datetime.strptime(day, "%Y%m%d").date()
                if "-" not in day
                else date.fromisoformat(day)
            )
        if day >= today:
            continue
        values = [float(bar.open), float(bar.high), float(bar.low), float(bar.close)]
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise HTTPException(502, "IBKR returned an invalid price bar.")
        opening, high, low, close = values
        if low > min(opening, close) or high < max(opening, close):
            raise HTTPException(502, "IBKR returned an inconsistent price bar.")
        result[day.isoformat()] = {
            "time": day.isoformat(),
            "open": opening,
            "high": high,
            "low": low,
            "close": close,
            "volume": max(0, float(bar.volume)),
        }
    return [result[key] for key in sorted(result)]


provider = IBKRProvider()


def normalize_intraday(bars, seconds, now=None):
    now = now or datetime.now(timezone.utc)
    result = {}
    for bar in bars:
        stamp = bar.date
        if not isinstance(stamp, datetime) or stamp.tzinfo is None:
            raise HTTPException(
                502, "IBKR intraday timestamps must include a timezone."
            )
        stamp = stamp.astimezone(timezone.utc)
        if stamp + timedelta(seconds=seconds) > now:
            continue
        values = [
            float(getattr(bar, name))
            for name in ("open", "high", "low", "close", "volume")
        ]
        if (
            not all(math.isfinite(v) for v in values)
            or min(values[:4]) <= 0
            or values[4] < 0
        ):
            raise HTTPException(502, "IBKR returned invalid intraday OHLCV.")
        opening, high, low, close, volume = values
        if low > min(opening, close) or high < max(opening, close):
            raise HTTPException(
                502, "IBKR returned an inconsistent intraday price bar."
            )
        stamp = stamp.isoformat().replace("+00:00", "Z")
        result[stamp] = dict(
            time=stamp, open=opening, high=high, low=low, close=close, volume=volume
        )
    return [result[k] for k in sorted(result)]


def normalize_ticks(ticks):
    result = []
    for tick in ticks:
        price, size = float(tick.price), float(tick.size)
        if (
            not math.isfinite(price)
            or price <= 0
            or not math.isfinite(size)
            or size < 0
        ):
            raise HTTPException(502, "IBKR returned an invalid trade tick.")
        result.append(
            {
                "time": tick.time.astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "price": price,
                "size": size,
                "exchange": tick.exchange,
                "conditions": tick.specialConditions,
                "past_limit": bool(tick.tickAttribLast.pastLimit),
                "unreported": bool(tick.tickAttribLast.unreported),
            }
        )
    return sorted(result, key=lambda row: row["time"])
