"""Optional user-Python adapter loaded only inside the native strategy worker.

All simulation, validation, statistics and reporting live in C++. This module
constructs the documented Python callback objects and loads a user's Strategy.
"""

from dataclasses import dataclass
import importlib.util
import json
from types import MappingProxyType


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
    history: tuple
    portfolio: Portfolio
    params: object

    @property
    def bar(self):
        return self.history[-1]


def load(path, bars_json, params_json):
    spec = importlib.util.spec_from_file_location("user_strategy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strategy = module.Strategy()
    history = tuple(Bar(**bar) for bar in json.loads(bars_json))
    params = MappingProxyType(json.loads(params_json))

    def on_bar(count, cash, shares, equity):
        return strategy.on_bar(
            Context(history[:count], Portfolio(cash, shares, equity), params)
        )

    return on_bar
