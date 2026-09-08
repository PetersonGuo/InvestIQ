import json
from datetime import date, datetime
from typing import Literal
from uuid import uuid4
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator
from api.trading import Ticker
from core.database import connect
from services import backtests, market

router = APIRouter(prefix="/api")


class StrategyInput(BaseModel):
    name: str = Field(default="My strategy", min_length=1, max_length=100)
    language: Literal["python", "cpp"]
    code: str = Field(min_length=1, max_length=64000)
    params: dict[str, float] = Field(default_factory=dict, max_length=50)

    @model_validator(mode="after")
    def finite_params(self):
        import math

        if not all(math.isfinite(value) for value in self.params.values()):
            raise ValueError("Strategy parameters must be finite numbers.")
        return self


class BacktestInput(StrategyInput):
    force_rerun: bool = False
    ticker: Ticker
    interval: str = "1d"
    start_date: str
    end_date: str
    initial_cash: float = Field(
        default=100000, ge=100, le=100000000, allow_inf_nan=False
    )
    commission: float = Field(default=1, ge=0, le=1000, allow_inf_nan=False)
    slippage_bps: float = Field(default=5, ge=0, le=1000, allow_inf_nan=False)
    warmup: int = Field(default=30, ge=0, le=250)

    @model_validator(mode="after")
    def dates(self):
        from services.resolutions import RESOLUTIONS, utc_time

        if self.interval not in RESOLUTIONS:
            raise ValueError("Unsupported candle interval.")
        if self.interval == "1d":
            self.start_date = date.fromisoformat(self.start_date).isoformat()
            self.end_date = date.fromisoformat(self.end_date).isoformat()
        else:
            self.start_date = (
                utc_time(self.start_date).isoformat().replace("+00:00", "Z")
            )
            self.end_date = utc_time(self.end_date).isoformat().replace("+00:00", "Z")
            if (
                utc_time(self.end_date) - utc_time(self.start_date)
            ).total_seconds() > 31 * 86400:
                raise ValueError(
                    "Intraday backtests support windows up to 31 calendar days. Small intervals may require shorter windows."
                )
        if (
            (self.start_date >= self.end_date)
            if self.interval == "1d"
            else (utc_time(self.start_date) >= utc_time(self.end_date))
        ):
            raise ValueError("Start date must be before end date.")
        if self.commission > self.initial_cash:
            raise ValueError("Commission cannot exceed initial cash.")
        return self


class ScannerInput(BaseModel):
    scan_code: Literal["TOP_PERC_GAIN", "TOP_PERC_LOSE", "HOT_BY_VOLUME"] = (
        "HOT_BY_VOLUME"
    )
    min_price: float = Field(default=5, ge=0, le=100000, allow_inf_nan=False)
    max_price: float = Field(default=1000, gt=0, le=100000, allow_inf_nan=False)
    min_volume: int = Field(default=100000, ge=0, le=1000000000)
    limit: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def prices(self):
        if self.min_price > self.max_price:
            raise ValueError("Minimum price cannot exceed maximum price.")
        return self


@router.get("/strategies/examples")
def examples():
    from simulation.presets import examples as strategy_examples

    return strategy_examples()


@router.get("/strategies")
def strategies():
    with connect() as db:
        rows = [
            dict(row)
            for row in db.execute("SELECT * FROM strategies ORDER BY updated_at DESC")
        ]
    for row in rows:
        row["params"] = json.loads(row.pop("params_json"))
    return rows


@router.post("/strategies", status_code=201)
def save_strategy(query: StrategyInput):
    strategy_id = str(uuid4())
    with connect() as db:
        db.execute(
            "INSERT INTO strategies VALUES (?,?,?,?,?,?)",
            (
                strategy_id,
                query.name,
                query.language,
                query.code,
                json.dumps(query.params),
                backtests.timestamp(),
            ),
        )
    return {"id": strategy_id, **query.model_dump()}


@router.post("/backtests", status_code=202)
def submit_backtest(query: BacktestInput):
    return backtests.submit(query.model_dump(mode="json"))


@router.get("/backtests")
def runs():
    with connect() as db:
        rows = [
            dict(row)
            for row in db.execute(
                "SELECT id,status,created_at,request_json,error FROM backtest_runs ORDER BY created_at DESC LIMIT 30"
            )
        ]
    for row in rows:
        request = json.loads(row.pop("request_json"))
        row.update({key: request[key] for key in ("name", "language", "ticker")})
    return rows


@router.get("/backtests/{run_id}")
def run(run_id: str):
    return backtests.get_run(run_id)


@router.post("/scanner")
def scan(query: ScannerInput):
    if market.DATA_MODE == "ibkr":
        from services.ibkr import provider

        rows = provider.request("scanner", tuple(query.model_dump().items()))
    elif market.DATA_MODE == "demo":
        items = [market.history(ticker, 2) for ticker in market.COMPANIES]
        items = [
            item
            for item in items
            if query.min_price <= item["price"] <= query.max_price
            and item["bars"][-1]["volume"] >= query.min_volume
        ]
        items.sort(
            key=lambda item: (
                item["bars"][-1]["volume"]
                if query.scan_code == "HOT_BY_VOLUME"
                else item["change_percent"]
            ),
            reverse=query.scan_code != "TOP_PERC_LOSE",
        )
        rows = [
            {
                "rank": index + 1,
                "ticker": item["ticker"],
                "name": item["name"],
                "exchange": "DEMO",
            }
            for index, item in enumerate(items[: query.limit])
        ]
    else:
        raise HTTPException(
            422,
            "Market scanning requires IBKR mode. Symbol search remains available with Massive.",
        )
    return {
        "results": rows,
        "source": market.DATA_MODE,
        "scan_code": query.scan_code,
        "as_of": backtests.timestamp(),
    }
