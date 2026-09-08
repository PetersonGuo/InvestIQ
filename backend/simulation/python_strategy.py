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


class HistoryWindow:
    def __init__(self, bars, count):
        self._bars, self._count = bars, count

    def __len__(self):
        return self._count

    def __getitem__(self, key):
        if isinstance(key, slice):
            return tuple(self._bars[i] for i in range(*key.indices(self._count)))
        index = key if key >= 0 else self._count + key
        if index < 0 or index >= self._count:
            raise IndexError("History contains completed bars only")
        return self._bars[index]


@dataclass(frozen=True)
class MultiPortfolio:
    cash: float
    equity: float
    positions: object


@dataclass(frozen=True)
class MultiContext:
    histories: object
    portfolio: MultiPortfolio
    params: object


def load_portfolio(path, assets_json, params_json):
    spec = importlib.util.spec_from_file_location("user_strategy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strategy = module.Strategy()
    assets = json.loads(assets_json)
    symbols = tuple(asset["ticker"] for asset in assets)
    histories = {
        asset["ticker"]: tuple(Bar(**bar) for bar in asset["bars"]) for asset in assets
    }
    params = MappingProxyType(json.loads(params_json))

    def on_bar(count, cash, equity, positions):
        result = strategy.on_bar(
            MultiContext(
                MappingProxyType(
                    {
                        symbol: HistoryWindow(histories[symbol], count)
                        for symbol in symbols
                    }
                ),
                MultiPortfolio(
                    cash, equity, MappingProxyType(dict(zip(symbols, positions)))
                ),
                params,
            )
        )
        if result is None:
            return None
        if not isinstance(result, dict) or any(
            symbol not in symbols for symbol in result
        ):
            raise ValueError(
                "Return None or a dict of symbol-to-target weights; omitted symbols target zero"
            )
        return [result.get(symbol, 0.0) for symbol in symbols]

    return on_bar
