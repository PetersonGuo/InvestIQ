"use client";

import Link from "next/link";
import { FormEvent, useCallback, useEffect, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Bell,
  ChartCandlestick,
  Check,
  RefreshCw,
  Search,
  Trash2,
  Wallet,
} from "lucide-react";
import { intervals } from "@/lib/intervals";
import TradeTape from "./trade-tape";
import KLineChart from "./kline-chart";
import { Alert, api, money, Portfolio, Stock } from "@/lib/stockassist";

type SearchResult = { ticker: string; name: string };
export default function Dashboard() {
  const [symbol, setSymbol] = useState("AAPL");
  const [interval, setIntervalResolution] = useState("1d");
  const [days, setDays] = useState(90);
  const [stock, setStock] = useState<Stock | null>(null);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [quantity, setQuantity] = useState("1");
  const [direction, setDirection] = useState<"above" | "below">("above");
  const [threshold, setThreshold] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  const [marketStatus, setMarketStatus] = useState<{
    provider: string;
    connected: boolean | null;
  } | null>(null);
  const [revision, setRevision] = useState(0);
  const refreshAccount = useCallback(async () => {
    const [p, a, m] = await Promise.all([
      api<Portfolio>("portfolio"),
      api<Alert[]>("alerts"),
      api<{ provider: string; connected: boolean | null }>("market/status"),
    ]);
    setPortfolio(p);
    setAlerts(a);
    setMarketStatus(m);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setStock(null);
    setError("");
    api<Stock>(`stocks/${encodeURIComponent(symbol)}?days=${days}&interval=${interval}`, {
      signal: controller.signal,
    })
      .then(setStock)
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [symbol, days, revision, interval]);

  useEffect(() => {
    let active = true;
    const refresh = () => {
      refreshAccount().catch((e) => {
        if (active) setError(e.message);
      });
    };
    refresh();
    const timer = setInterval(refresh, 15000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [refreshAccount]);

  useEffect(() => {
    const controller = new AbortController();
    if (!query.trim()) {
      setResults([]);
      setSearching(false);
      return () => controller.abort();
    }
    setSearching(true);
    const timer = setTimeout(() => {
      api<{ results: SearchResult[] }>(
        `search?ticker=${encodeURIComponent(query)}`,
        { signal: controller.signal },
      )
        .then((data) => setResults(data.results))
        .catch((e) => {
          if (!controller.signal.aborted) setError(e.message);
        })
        .finally(() => {
          if (!controller.signal.aborted) setSearching(false);
        });
    }, 250);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  function selectTicker(ticker: string) {
    setSymbol(ticker);
    setQuery("");
    setThreshold("");
    setEditing(null);
    setNotice("");
  }
  async function mutate(action: () => Promise<unknown>, message: string) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await action();
      setNotice(message);
      await refreshAccount();
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Something went wrong. Please retry.",
      );
    } finally {
      setBusy(false);
    }
  }
  function submitOrder(event: FormEvent) {
    event.preventDefault();
    void mutate(
      () =>
        api("order", {
          method: "POST",
          body: JSON.stringify({
            ticker: symbol,
            side,
            quantity: Number(quantity),
          }),
        }),
      `Paper ${side} filled for ${quantity} ${symbol} shares.`,
    );
  }
  function submitAlert(event: FormEvent) {
    event.preventDefault();
    void mutate(async () => {
      await api(editing ? `alerts/${editing}` : "alerts", {
        method: editing ? "PUT" : "POST",
        body: JSON.stringify({
          ticker: symbol,
          direction,
          threshold: Number(threshold),
        }),
      });
      setThreshold("");
      setEditing(null);
    }, "Price alert saved. It will be checked within 30 seconds.");
  }
  const up = (stock?.change_percent ?? 0) >= 0;
  const shares =
    portfolio?.positions.find((p) => p.ticker === symbol)?.quantity ?? 0;
  const inputClass =
    "w-full rounded-lg border border-[#2b3948] bg-[#0c141e] px-3 py-2.5 text-sm text-slate-100 outline-none focus:border-emerald-400";
  const buttonClass =
    "rounded-lg bg-emerald-400 px-4 py-2.5 text-sm font-semibold text-[#09251e] transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-40";

  return (
    <main className="min-h-screen bg-[#0a111a] text-[#e2eaf3] selection:bg-emerald-800">
      <header className="border-b border-[#233040] bg-[#0d1620]">
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-4 px-5 py-5 lg:px-10">
          <Link
            href="/"
            className="flex items-center gap-3 text-xl font-semibold tracking-tight"
          >
            <span className="rounded-xl bg-emerald-400/10 p-2 text-emerald-400">
              <ChartCandlestick size={24} />
            </span>
            StockAssist
            <span className="ml-2 hidden rounded border border-[#344252] px-2 py-0.5 text-[10px] font-medium tracking-widest text-slate-400 sm:inline">
              WORKSPACE
            </span>
          </Link>
          <Link
            href="/research"
            className="text-sm text-emerald-300 hover:text-emerald-200"
          >
            Research & backtest
          </Link>
          <Link
            href="/pairs"
            className="text-sm text-emerald-300 hover:text-emerald-200"
          >
            Pair alerts
          </Link>
          <div className="flex items-center gap-4 text-xs text-slate-400">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            <span>
              {marketStatus?.provider === "ibkr"
                ? `IBKR ${marketStatus.connected ? "connected" : "disconnected"}`
                : marketStatus?.provider === "demo"
                  ? "Demo data"
                  : "Historical data"}{" "}
              · Paper account
            </span>{" "}
            <span className="rounded-full bg-[#203043] px-3 py-2 text-slate-200">
              SA
            </span>
          </div>
        </div>
      </header>
      <div className="mx-auto max-w-[1440px] px-5 py-8 lg:px-10">
        <div className="mb-8 flex flex-wrap items-end justify-between gap-5">
          <div>
            <p className="mb-2 text-xs font-medium uppercase tracking-[.2em] text-emerald-400">
              A clearer view of the market
            </p>
            <h1 className="text-3xl font-semibold tracking-tight">
              Market overview
            </h1>
            <p className="mt-2 text-sm text-slate-400">
              Explore a stock. Set your levels. Practice your next move.
            </p>
          </div>
          <div className="relative w-full sm:w-80">
            <label htmlFor="stock-search" className="sr-only">
              Search stocks
            </label>
            <Search
              className="absolute left-3 top-3 text-slate-500"
              size={18}
            />
            <input
              id="stock-search"
              className={`${inputClass} pl-10`}
              placeholder="Search symbol or company…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              autoComplete="off"
            />
            {query.trim() && (
              <div
                className="absolute z-20 mt-2 max-h-80 w-full overflow-auto rounded-xl border border-[#2b3948] bg-[#142130] p-2 shadow-xl"
                aria-live="polite"
              >
                {searching ? (
                  <p className="p-3 text-sm text-slate-400">Searching…</p>
                ) : results.length ? (
                  results.map((r) => (
                    <button
                      key={r.ticker}
                      className="flex w-full items-center justify-between gap-3 rounded-lg p-3 text-left text-sm hover:bg-[#243449]"
                      onClick={() => selectTicker(r.ticker)}
                    >
                      <b>{r.ticker}</b>
                      <span className="truncate text-xs text-slate-400">
                        {r.name}
                      </span>
                    </button>
                  ))
                ) : (
                  <p className="p-3 text-sm text-slate-400">
                    No matching stocks.
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
        <div className="mb-6 flex flex-wrap gap-2">
          {["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "SPY"].map(
            (t) => (
              <button
                key={t}
                onClick={() => selectTicker(t)}
                aria-pressed={t === symbol}
                className={`rounded-full border px-4 py-2 text-xs font-medium transition ${t === symbol ? "border-emerald-400/50 bg-emerald-400/10 text-emerald-300" : "border-[#263343] text-slate-400 hover:border-slate-500 hover:text-white"}`}
              >
                {t}
              </button>
            ),
          )}
        </div>
        {error && (
          <div
            role="alert"
            className="mb-5 flex items-center justify-between gap-3 rounded-xl border border-red-400/30 bg-red-400/10 p-4 text-sm text-red-200"
          >
            {error}
            <button
              className="underline"
              onClick={() => {
                setRevision((n) => n + 1);
                void refreshAccount().catch((e) => setError(e.message));
              }}
            >
              Retry
            </button>
          </div>
        )}
        {notice && (
          <div
            role="status"
            className="mb-5 flex items-center gap-2 rounded-xl border border-emerald-400/20 bg-emerald-400/10 p-4 text-sm text-emerald-200"
          >
            <Check size={16} />
            {notice}
          </div>
        )}
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_330px]">
          <div className="min-w-0 space-y-6">
            <section className="overflow-hidden rounded-2xl border border-[#243141] bg-[#101923]">
              <div className="flex flex-wrap justify-between gap-4 p-6">
                <div className="flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[#223044] text-sm font-bold">
                    {symbol.slice(0, 2)}
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold">{symbol}</h2>
                    <p className="text-sm text-slate-400">
                      {stock?.name ?? "Stock price history"}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-3xl font-semibold tracking-tight">
                    {stock ? money(stock.price) : "—"}
                  </p>
                  {stock && (
                    <p
                      className={`mt-1 flex items-center justify-end gap-1 text-sm ${up ? "text-emerald-400" : "text-red-300"}`}
                    >
                      {up ? (
                        <ArrowUpRight size={16} />
                      ) : (
                        <ArrowDownRight size={16} />
                      )}
                      {up ? "+" : ""}
                      {stock.change_percent.toFixed(2)}%{" "}
                      <span className="ml-1 text-xs text-slate-500">
                        vs. prior bar
                      </span>
                    </p>
                  )}
                </div>
              </div>
              <div className="flex flex-wrap items-center justify-between gap-3 border-y border-[#243141] px-6 py-3">
                <span className="text-xs text-slate-400">
                  <label>Candle interval <select aria-label="Candle interval" value={interval} onChange={e => setIntervalResolution(e.target.value)} className="ml-2 rounded bg-[#223044] p-2 text-white">{intervals.map(value => <option key={value} value={value}>{value === "1d" ? "Daily" : value}</option>)}</select></label> · USD {interval !== "1d" && "· UTC"}
                </span>
                <div className="flex gap-1">
                  {(interval === "1d" ? [
                    [20, "1M"],
                    [60, "3M"],
                    [90, "4M"],
                    [250, "1Y"],
                  ] : []).map(([n, label]) => (
                    <button
                      key={n}
                      onClick={() => setDays(Number(n))}
                      aria-pressed={days === n}
                      className={`rounded px-3 py-1 text-xs ${days === n ? "bg-[#2a3b4e] text-white" : "text-slate-500 hover:text-white"}`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
              {loading ? (
                <div
                  className="flex h-[360px] items-center justify-center gap-2 text-sm text-slate-400"
                  role="status"
                >
                  <RefreshCw size={16} className="animate-spin" />
                  Loading price history…
                </div>
              ) : stock ? (
                <KLineChart symbol={symbol} bars={stock.bars} interval={interval} />
              ) : (
                <div className="flex h-[360px] items-center justify-center text-sm text-slate-400">
                  Price history unavailable.
                </div>
              )}
              <div className="border-t border-[#243141] px-6 py-4 text-xs text-slate-400">
                {stock
                  ? `${stock.source === "demo" ? "SYNTHETIC DEMO DATA" : stock.source === "ibkr" ? "IBKR · HISTORICAL DATA" : "MASSIVE · HISTORICAL DATA"} · Last bar ${stock.as_of}. `
                  : ""}
                Prices are not live quotes.
              </div>
            </section>
            <TradeTape symbol={symbol} />
            <section className="rounded-2xl border border-[#243141] bg-[#101923] p-6">
              <div className="mb-5 flex items-center justify-between">
                <h2 className="flex items-center gap-2 font-semibold">
                  <Bell size={18} className="text-emerald-400" />
                  Price alerts
                </h2>
                <span className="text-xs text-slate-500">
                  {alerts.filter((a) => a.active).length} active
                </span>
              </div>
              {alerts.length === 0 ? (
                <p className="py-7 text-center text-sm text-slate-500">
                  Your levels, on watch. Create your first price alert.
                </p>
              ) : (
                <ul className="divide-y divide-[#243141]">
                  {alerts.map((a) => (
                    <li
                      key={a.id}
                      className="flex flex-wrap items-center justify-between gap-3 py-4"
                    >
                      <div>
                        <button
                          onClick={() => selectTicker(a.ticker)}
                          className="text-sm font-semibold hover:text-emerald-400"
                        >
                          {a.ticker}
                        </button>
                        <p className="mt-1 text-xs text-slate-400">
                          {a.direction === "above"
                            ? "At or above"
                            : "At or below"}{" "}
                          {money(a.threshold)}
                        </p>
                      </div>
                      <div className="flex items-center gap-3">
                        <span
                          className={`rounded-full px-2 py-1 text-[11px] ${a.active ? "bg-blue-400/10 text-blue-300" : "bg-emerald-400/10 text-emerald-300"}`}
                        >
                          {a.active ? "Watching" : "Triggered"}
                        </span>
                        <button
                          disabled={busy}
                          className="text-xs text-slate-400 hover:text-white"
                          onClick={() => {
                            setSymbol(a.ticker);
                            setDirection(a.direction);
                            setThreshold(String(a.threshold));
                            setEditing(a.id);
                          }}
                        >
                          {a.active ? "Edit" : "Rearm"}
                        </button>
                        <button
                          disabled={busy}
                          aria-label={`Delete ${a.ticker} alert`}
                          onClick={() =>
                            void mutate(
                              () => api(`alerts/${a.id}`, { method: "DELETE" }),
                              "Alert removed.",
                            )
                          }
                          className="p-2 text-slate-500 hover:text-red-300"
                        >
                          <Trash2 size={15} />
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
              <p className="mt-4 text-xs leading-relaxed text-slate-500">
                One-shot alerts check the latest daily close every 30 seconds
                while the backend runs. Triggered alerts appear here; email and
                push delivery are not enabled.
              </p>
            </section>
          </div>
          <aside className="space-y-6">
            <section className="rounded-2xl border border-[#243141] bg-gradient-to-br from-[#142b2b] to-[#101923] p-6">
              <h2 className="flex items-center gap-2 text-sm text-slate-300">
                <Wallet size={17} className="text-emerald-400" />
                Paper buying power
              </h2>
              <p className="mt-4 text-3xl font-semibold">
                {portfolio ? money(portfolio.cash_cents / 100) : "—"}
              </p>
              <p className="mt-2 text-xs text-slate-400">
                $100,000 starting balance · USD
              </p>
            </section>
            <section className="rounded-2xl border border-[#243141] bg-[#101923] p-6">
              <h2 className="mb-5 font-semibold">Practice a trade</h2>
              <form onSubmit={submitOrder} className="space-y-4">
                <div className="flex rounded-lg bg-[#0a111a] p-1">
                  {(["buy", "sell"] as const).map((s) => (
                    <button
                      key={s}
                      type="button"
                      aria-pressed={side === s}
                      onClick={() => setSide(s)}
                      className={`w-1/2 rounded-md py-2 text-sm capitalize ${side === s ? "bg-[#263747] text-white" : "text-slate-500"}`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
                <div>
                  <label
                    htmlFor="quantity"
                    className="mb-2 block text-xs text-slate-400"
                  >
                    Shares of {symbol}
                  </label>
                  <input
                    id="quantity"
                    className={inputClass}
                    type="number"
                    min="1"
                    max="1000000"
                    step="1"
                    required
                    value={quantity}
                    onChange={(e) => setQuantity(e.target.value)}
                  />
                </div>
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Estimated total</span>
                  <span className="text-slate-200">
                    {interval !== "1d" ? "Daily-close price at fill" : stock ? money(stock.price * Number(quantity)) : "—"}
                  </span>
                </div>
                <button
                  disabled={busy || !stock || !portfolio}
                  className={`${buttonClass} w-full`}
                >
                  {busy
                    ? "Working…"
                    : `${side === "buy" ? "Buy" : "Sell"} ${symbol} · Paper`}
                </button>
                <p className="text-xs leading-relaxed text-slate-500">
                  {shares} shares held. Fills use the latest available daily
                  close. No real money or brokerage connection.
                </p>
              </form>
            </section>
            <section className="rounded-2xl border border-[#243141] bg-[#101923] p-6">
              <h2 className="mb-5 font-semibold">
                {editing ? "Edit & rearm alert" : "Create price alert"}
              </h2>
              <form onSubmit={submitAlert} className="space-y-4">
                <div>
                  <label
                    htmlFor="direction"
                    className="mb-2 block text-xs text-slate-400"
                  >
                    Notify me when {symbol} closes
                  </label>
                  <select
                    id="direction"
                    className={inputClass}
                    value={direction}
                    onChange={(e) =>
                      setDirection(e.target.value as "above" | "below")
                    }
                  >
                    <option value="above">At or above</option>
                    <option value="below">At or below</option>
                  </select>
                </div>
                <div>
                  <label
                    htmlFor="threshold"
                    className="mb-2 block text-xs text-slate-400"
                  >
                    Target price · USD
                  </label>
                  <input
                    id="threshold"
                    className={inputClass}
                    type="number"
                    min="0.01"
                    max="10000000"
                    step="0.01"
                    placeholder={stock?.price.toFixed(2) ?? "0.00"}
                    value={threshold}
                    onChange={(e) => setThreshold(e.target.value)}
                    required
                  />
                </div>
                <button
                  disabled={busy || !stock}
                  className="w-full rounded-lg border border-[#3c5266] px-4 py-2.5 text-sm font-medium hover:bg-[#223044] disabled:opacity-40"
                >
                  {editing ? "Save alert" : "Create alert"}
                </button>
                {editing && (
                  <button
                    type="button"
                    className="w-full text-xs text-slate-400"
                    onClick={() => {
                      setEditing(null);
                      setThreshold("");
                    }}
                  >
                    Cancel editing
                  </button>
                )}
              </form>
            </section>
          </aside>
        </div>
        <section className="mt-6 overflow-hidden rounded-2xl border border-[#243141] bg-[#101923]">
          <h2 className="flex items-center gap-2 px-6 pt-6 font-semibold">
            <Activity size={18} className="text-emerald-400" />
            Your paper portfolio
          </h2>
          <div className="grid gap-8 p-6 md:grid-cols-2">
            <div>
              <h3 className="mb-4 text-xs uppercase tracking-widest text-slate-500">
                Open positions
              </h3>
              {portfolio?.positions.length ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="text-xs text-slate-500">
                      <tr>
                        <th className="pb-3 font-normal">Symbol</th>
                        <th className="pb-3 font-normal">Shares</th>
                        <th className="pb-3 text-right font-normal">
                          Cost basis
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {portfolio.positions.map((p) => (
                        <tr
                          key={p.ticker}
                          className="border-t border-[#243141]"
                        >
                          <td className="py-3">
                            <button
                              onClick={() => selectTicker(p.ticker)}
                              className="font-medium hover:text-emerald-400"
                            >
                              {p.ticker}
                            </button>
                          </td>
                          <td>{p.quantity}</td>
                          <td className="text-right">
                            {money(p.cost_cents / 100)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="py-4 text-sm text-slate-500">
                  Your first paper trade will appear here.
                </p>
              )}
            </div>
            <div>
              <h3 className="mb-4 text-xs uppercase tracking-widest text-slate-500">
                Recent fills
              </h3>
              {portfolio?.orders.length ? (
                <ul className="max-h-72 divide-y divide-[#243141] overflow-auto">
                  {portfolio.orders.map((o) => (
                    <li
                      key={o.id}
                      className="flex justify-between py-3 text-sm"
                    >
                      <div>
                        <span
                          className={
                            o.side === "buy"
                              ? "text-emerald-400"
                              : "text-orange-300"
                          }
                        >
                          {o.side.toUpperCase()}
                        </span>{" "}
                        {o.quantity} {o.ticker}
                        <p className="mt-1 text-[11px] text-slate-500">
                          {new Date(o.created_at).toLocaleString()} · close{" "}
                          {o.price_as_of}
                        </p>
                      </div>
                      <span>{money(o.price_cents / 100)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="py-4 text-sm text-slate-500">
                  No trades yet. Start with a stock you know.
                </p>
              )}
            </div>
          </div>
        </section>
        <footer className="mt-8 flex flex-wrap justify-between gap-3 text-xs text-slate-600">
          <span>StockAssist · A little more perspective.</span>
          <span>Local workspace · Historical prices · Simulated trades</span>
        </footer>
      </div>
    </main>
  );
}
