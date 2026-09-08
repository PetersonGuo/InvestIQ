"""A daily, long-only, single-asset event-driven backtester.

Strategies see completed bars only. A close-time target is filled at the NEXT
bar's open. Holdings remaining at the end are marked to the last close.
"""

from dataclasses import dataclass
from math import floor, isfinite, sqrt, expm1, log
from statistics import pstdev, mean
from types import MappingProxyType
from typing import Callable


@dataclass(frozen=True)
class Bar:
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Portfolio:
    cash: float
    shares: int
    equity: float


@dataclass(frozen=True)
class Context:
    history: tuple[Bar, ...]
    portfolio: Portfolio
    params: dict

    @property
    def bar(self):
        return self.history[-1]


def simulate(
    bars,
    callback: Callable,
    params=None,
    initial_cash=100000.0,
    commission=1.0,
    slippage_bps=5.0,
    warmup=0,
):
    if len(bars) < 2 or not 0 <= warmup <= len(bars) - 2:
        raise ValueError("Choose at least two trading bars after warmup.")
    if not isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("Initial cash must be positive.")
    if not 0 <= commission <= initial_cash or not 0 <= slippage_bps <= 1000:
        raise ValueError("Invalid commission or slippage.")
    history = tuple(Bar(**row) for row in bars)
    for index, bar in enumerate(history):
        values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
        if (
            not all(isfinite(value) for value in values)
            or min(values[:4]) <= 0
            or bar.volume < 0
        ):
            raise ValueError("Invalid OHLCV data.")
        if bar.low > min(bar.open, bar.close) or bar.high < max(bar.open, bar.close):
            raise ValueError("Invalid OHLC range.")
        if index and bar.time <= history[index - 1].time:
            raise ValueError("Bars must have unique, ascending dates.")
    cash, shares, cost = float(initial_cash), 0, 0.0
    pending = None
    fills, curve, closed_pnl = [], [], []
    peak = initial_cash
    max_drawdown = 0.0
    fees = 0.0
    benchmark_shares = 0
    benchmark_cash = initial_cash
    params = MappingProxyType(params or {})
    for index, bar in enumerate(history):
        if index == warmup + 1:
            price = bar.open * (1 + slippage_bps / 10000)
            benchmark_shares = max(0, floor((initial_cash - commission) / price))
            benchmark_cash = (
                initial_cash
                - benchmark_shares * price
                - (commission if benchmark_shares else 0)
            )
        if pending is not None and index > warmup:
            weight, signal_date = pending
            equity_at_open = cash + shares * bar.open
            target = floor(equity_at_open * weight / bar.open)
            delta = target - shares
            if delta > 0:
                price = bar.open * (1 + slippage_bps / 10000)
                quantity = min(delta, max(0, floor((cash - commission) / price)))
                if quantity:
                    total = quantity * price + commission
                    cash -= total
                    cost += total
                    shares += quantity
                    fees += commission
                    fills.append(
                        {
                            "signal_date": signal_date,
                            "time": bar.time,
                            "side": "buy",
                            "quantity": quantity,
                            "price": round(price, 6),
                            "commission": commission,
                            "realized_pnl": None,
                        }
                    )
            elif delta < 0:
                price = bar.open * (1 - slippage_bps / 10000)
                quantity = min(-delta, shares)
                proceeds = quantity * price - commission
                if proceeds >= 0:
                    basis = cost * quantity / shares
                    pnl = proceeds - basis
                    closed_pnl.append(pnl)
                    cash += proceeds
                    cost -= basis
                    shares -= quantity
                    fees += commission
                    fills.append(
                        {
                            "signal_date": signal_date,
                            "time": bar.time,
                            "side": "sell",
                            "quantity": quantity,
                            "price": round(price, 6),
                            "commission": commission,
                            "realized_pnl": round(pnl, 6),
                        }
                    )
            pending = None
        equity = cash + shares * bar.close
        if index >= warmup:
            peak = max(peak, equity)
            drawdown = (equity / peak - 1) * 100
            max_drawdown = min(max_drawdown, drawdown)
            curve.append(
                {
                    "time": bar.time,
                    "equity": round(equity, 6),
                    "benchmark": round(
                        benchmark_cash + benchmark_shares * bar.close, 6
                    ),
                    "drawdown_percent": round(drawdown, 6),
                    "cash": round(cash, 6),
                    "shares": shares,
                }
            )
        # Warmup callbacks can build indicators/state, but cannot place orders.
        context = Context(history[: index + 1], Portfolio(cash, shares, equity), params)
        weight = callback(context)
        if weight is not None:
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (float, int))
                or not isfinite(weight)
                or not 0 <= weight <= 1
            ):
                raise ValueError(
                    f"Strategy returned {weight!r} on {bar.time}; expected None or a target weight from 0 to 1."
                )
            if index >= warmup:
                pending = (float(weight), bar.time)
    return summarize(
        curve, fills, closed_pnl, initial_cash, max_drawdown, fees, shares, pending
    )


def summarize(
    curve, fills, closed_pnl, initial_cash, max_drawdown, fees, shares, pending
):
    values = [point["equity"] for point in curve]
    returns = [b / a - 1 for a, b in zip(values, values[1:])]
    volatility = pstdev(returns) if len(returns) > 1 else 0
    years = (len(curve) - 1) / 252
    annualized = None
    if years > 0 and values[-1] > 0:
        exponent = log(values[-1] / initial_cash) / years
        if exponent < 700:
            annualized = expm1(exponent) * 100
    result = {
        "metrics": {
            "initial_cash": initial_cash,
            "final_equity": round(values[-1], 6),
            "total_return_percent": (values[-1] / initial_cash - 1) * 100,
            "benchmark_return_percent": (curve[-1]["benchmark"] / initial_cash - 1)
            * 100,
            "annualized_return_percent": annualized,
            "max_drawdown_percent": abs(max_drawdown),
            "sharpe_ratio": (
                mean(returns) / volatility * sqrt(252) if volatility else None
            ),
            "fill_count": len(fills),
            "sell_count": len(closed_pnl),
            "win_rate_percent": (
                100 * sum(pnl > 0 for pnl in closed_pnl) / len(closed_pnl)
                if closed_pnl
                else None
            ),
            "total_commission": fees,
            "open_shares": shares,
        },
        "equity_curve": curve,
        "fills": fills,
        "unfilled_final_signal": pending[0] if pending else None,
    }
    return result
