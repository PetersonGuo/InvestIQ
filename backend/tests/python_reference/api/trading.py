from datetime import date, datetime, timezone
from typing import Annotated, Literal
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, StringConstraints
from core.database import connect
from services import market, trade

router = APIRouter(prefix="/api")
Ticker = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, to_upper=True, pattern=r"^[A-Z][A-Z0-9.\-]{0,14}$"
    ),
]


class TickerQuery(BaseModel):
    ticker: str = Field(default="", max_length=100)
    market: Literal["stocks"] = "stocks"
    limit: int = Field(default=20, ge=1, le=100)


class AlertQuery(BaseModel):
    ticker: Ticker
    direction: Literal["above", "below"]
    threshold: float = Field(gt=0, le=10000000, allow_inf_nan=False)


class OrderQuery(BaseModel):
    ticker: Ticker
    side: Literal["buy", "sell"]
    quantity: int = Field(gt=0, le=1000000, strict=True)


@router.get("/search")
def search(
    ticker: str = Query("", max_length=100), limit: int = Query(20, ge=1, le=100)
):
    return {"results": market.search(ticker, limit), "source": market.DATA_MODE}


@router.post("/search")
def search_post(query: TickerQuery):
    return search(query.ticker, query.limit)


@router.get("/stocks/{ticker}")
def stock(
    ticker: Ticker,
    days: int = Query(90, ge=2, le=250),
    before: str | None = Query(None, max_length=40),
    interval: str = "1d",
):
    from services.resolutions import validate_interval, utc_time

    validate_interval(interval)
    if before:
        try:
            before = (
                date.fromisoformat(before).isoformat()
                if interval == "1d"
                else utc_time(before).isoformat().replace("+00:00", "Z")
            )
        except ValueError:
            raise HTTPException(
                422,
                "Use an ISO date for daily data, or an ISO timestamp for intraday data.",
            )
    if interval != "1d":
        return market.history(ticker, days, before, interval)
    return (
        market.history(ticker, days, before) if before else market.history(ticker, days)
    )


@router.get("/stocks/{ticker}/ticks")
def ticks(ticker: Ticker, before: datetime | None = None):
    return market.trade_ticks(ticker, before.isoformat() if before else None)


@router.get("/alerts")
@router.post("/alert/list")
def list_alerts():
    with connect() as db:
        return [
            dict(row)
            for row in db.execute("SELECT * FROM alerts ORDER BY created_at DESC")
        ]


@router.post("/alerts", status_code=201)
@router.post("/alert/add", status_code=201)
def add_alert(query: AlertQuery):
    market.history(query.ticker, 2)
    alert = {
        "id": str(uuid4()),
        **query.model_dump(),
        "active": 1,
        "triggered_at": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with connect() as db:
        db.execute(
            "INSERT INTO alerts VALUES (:id,:ticker,:direction,:threshold,:active,:triggered_at,:created_at)",
            alert,
        )
    return alert


@router.put("/alerts/{alert_id}")
def update_alert(alert_id: str, query: AlertQuery):
    market.history(query.ticker, 2)
    with connect() as db:
        result = db.execute(
            "UPDATE alerts SET ticker=?,direction=?,threshold=?,active=1,triggered_at=NULL WHERE id=?",
            (query.ticker, query.direction, query.threshold, alert_id),
        )
        if result.rowcount == 0:
            raise HTTPException(404, "Alert not found.")
        return dict(
            db.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        )


@router.delete("/alerts/{alert_id}", status_code=204)
def remove_alert(alert_id: str):
    with connect() as db:
        if db.execute("DELETE FROM alerts WHERE id=?", (alert_id,)).rowcount == 0:
            raise HTTPException(404, "Alert not found.")


@router.get("/portfolio")
def portfolio():
    return trade.portfolio()


@router.post("/order", status_code=201)
def place_order(query: OrderQuery):
    return trade.place_order(query)
