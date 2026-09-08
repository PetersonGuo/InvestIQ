from typing import Literal
from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator
from api.trading import Ticker
from services import pairs

router = APIRouter(prefix="/api/pairs", tags=["pair alerts"])


class PairConfig(BaseModel):
    ticker_a: Ticker
    ticker_b: Ticker
    metric: Literal["zscore", "ratio"] = "zscore"
    condition: Literal["above", "below", "outside", "inside"] = "outside"
    threshold: float = Field(default=2, ge=-100000, le=100000, allow_inf_nan=False)
    lookback: int = Field(default=60, ge=20, le=250)
    hedge_ratio: float = Field(default=1, gt=0, le=100, allow_inf_nan=False)
    repeat: bool = False

    @model_validator(mode="after")
    def validate_pair(self):
        if self.ticker_a == self.ticker_b:
            raise ValueError("Choose two different stocks.")
        if self.metric == "ratio":
            if self.condition not in {"above", "below"} or self.threshold <= 0:
                raise ValueError(
                    "Ratio alerts require above/below and a positive threshold."
                )
        elif abs(self.threshold) > 20:
            raise ValueError("Z-score threshold must be between -20 and 20.")
        if self.condition in {"inside", "outside"} and self.threshold < 0:
            raise ValueError("Band thresholds cannot be negative.")
        return self


class ActiveState(BaseModel):
    active: bool


@router.post("/preview")
def preview(query: PairConfig):
    return pairs.analyze(query.model_dump())


@router.get("/alerts")
def list_alerts():
    return pairs.list_alerts()


@router.post("/alerts", status_code=201)
def create(query: PairConfig):
    return pairs.save(query.model_dump())


@router.put("/alerts/{alert_id}")
def update(alert_id: str, query: PairConfig):
    return pairs.save(query.model_dump(), alert_id)


@router.post("/alerts/{alert_id}/state")
def state(alert_id: str, query: ActiveState):
    return pairs.set_active(alert_id, query.active)


@router.delete("/alerts/{alert_id}", status_code=204)
def remove(alert_id: str):
    pairs.delete(alert_id)


@router.get("/events")
def events():
    return pairs.events()
