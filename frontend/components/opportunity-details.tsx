"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/stockassist";
import { Opportunity } from "@/lib/stock-lists";

type Metric = {
  label: string;
  value: number | null;
  unit: string;
  basis: string;
  period_end?: string;
  filed?: string;
  filing_url?: string;
};
type Facts = {
  ticker: string;
  name?: string;
  source: string;
  fetched_at?: string;
  metrics: Metric[];
  notes: string[];
};
type Run = {
  id: string;
  status: string;
  error?: string;
  result?: {
    risk_comparison: Record<
      string,
      {
        total_return_percent: number;
        max_drawdown_percent: number;
        volatility_percent: number;
        beta_to_spy: number | null;
        time_in_market_percent: number;
      }
    >;
    fills: unknown[];
    strategy_diagnostics?: Record<string, number>;
  };
};
const button =
  "rounded-lg border border-slate-600 px-3 py-2 text-sm disabled:opacity-40";
const format = (value: number) =>
  value.toLocaleString(undefined, {
    maximumFractionDigits: 2,
    notation: Math.abs(value) >= 1e6 ? "compact" : "standard",
  });
export default function OpportunityDetails({ row }: { row: Opportunity }) {
  const [compare, setCompare] = useState(
    row.relative_value.references[0]?.ticker ?? row.ticker_b,
  );
  const [facts, setFacts] = useState<Facts[]>([]);
  const [error, setError] = useState("");
  const [run, setRun] = useState<Run | null>(null);
  const [busy, setBusy] = useState(false);
  const snapshot = row.latest;
  useEffect(() => {
    const controller = new AbortController();
    setFacts([]);
    setError("");
    void Promise.all(
      [row.ticker_a, compare].map(async (ticker) => {
        try {
          return await api<Facts>(`stocks/${ticker}/fundamentals`, {
            signal: controller.signal,
          });
        } catch (e) {
          return {
            ticker,
            source: "unavailable",
            metrics: [],
            notes: [
              e instanceof Error ? e.message : "Fundamentals unavailable.",
            ],
          };
        }
      }),
    ).then((result) => {
      if (!controller.signal.aborted) setFacts(result);
    });
    return () => controller.abort();
  }, [row.ticker_a, compare]);
  useEffect(() => {
    if (!run || !["queued", "running"].includes(run.status)) return;
    let alive = true;
    const timer = setTimeout(() => {
      void api<Run>(`backtests/${run.id}`)
        .then((value) => {
          if (alive) setRun(value);
        })
        .catch((e) => {
          if (alive) {
            setError(e.message);
            setRun((old) =>
              old ? { ...old, status: "status unavailable" } : old,
            );
          }
        });
    }, 1500);
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [run]);
  const reason =
    row.discovery?.reason ??
    {
      competitor:
        "The model estimates a reference-stock hedge and removes the chosen market, sector and factor return exposures.",
      basket:
        "The target is modeled using the reference basket and exposure hedges.",
      same_company:
        "This configured share-class relationship compares securities of the same issuer; voting rights can sustain a spread.",
      etf: "This configured ETF relationship compares similar underlying exposure; expenses and tracking differences remain.",
      lead_lag:
        "The model forecasts the next daily residual from lagged reference residuals. It is not a low-latency execution model.",
      cross_asset:
        "This model uses listed stock/ETF proxies for a cross-asset relationship.",
      event:
        "The model combines supplied event terms with an exposure-adjusted divergence signal.",
    }[row.relative_value.category];
  return (
    <div className="mt-4 space-y-5 text-sm">
      <div>
        <h4 className="font-semibold">Why this setup?</h4>
        <p className="mt-1 text-slate-300">{reason}</p>
        {row.discovery && (
          <p className="mt-1 text-xs text-slate-400">
            Selected from {row.discovery.list_name}, list version{" "}
            {row.discovery.list_version}. This is the selection context; the
            current signal is refitted separately.
          </p>
        )}
      </div>
      {snapshot && (
        <div className="space-y-2">
          <h4 className="font-semibold">What is the model seeing?</h4>
          <p>
            The {snapshot.relative_value.score_type} is{" "}
            <b>{snapshot.value.toFixed(3)}</b>, against an entry magnitude of{" "}
            <b>{row.action.entry_z}</b>.{" "}
            {row.relative_value.category === "lead_lag"
              ? "The sign indicates the forecast direction."
              : snapshot.value > 0
                ? "The target-side residual is above its estimated baseline; the model considers selling that side."
                : "The target-side residual is below its estimated baseline; the model considers buying that side."}
          </p>
          <p className="text-slate-400">
            {snapshot.source} completed daily data: {snapshot.as_of}.
            Estimation: {snapshot.baseline_start}–{snapshot.baseline_end}.
            In-sample R² {snapshot.relative_value.fit_r2.toFixed(3)}. This
            regression does not prove cointegration or mispricing.
          </p>
          <p className="text-slate-400">
            Backtest exit band: |score| ≤ {row.action.exit_z}; stop band:
            |score| ≥ {row.action.stop_z}. The app does not track whether you
            opened this trade or manage a position-specific exit.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <caption className="mb-2 text-left font-semibold">
                Exposure removed by the hedge legs
              </caption>
              <thead>
                <tr>
                  <th className="p-2">Proxy</th>
                  <th>Role</th>
                  <th>Before β</th>
                  <th>After β</th>
                </tr>
              </thead>
              <tbody>
                {snapshot.relative_value.exposures.map((e) => (
                  <tr key={e.ticker} className="border-t border-slate-800">
                    <td className="p-2">{e.ticker}</td>
                    <td>{e.role}</td>
                    <td>{e.before_beta.toFixed(4)}</td>
                    <td>{e.after_beta.toFixed(6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-slate-400">
            {snapshot.relative_value.neutrality}
          </p>
          {row.action.state !== "entry" && (
            <details className="rounded border border-slate-700 p-2">
              <summary className="cursor-pointer">
                Analytical weights — no trade to open now
              </summary>
              <ul className="mt-2 space-y-1">
                {snapshot.relative_value.hedge_legs.map((leg) => (
                  <li key={leg.ticker}>
                    {leg.side === "long" ? "Long" : "Short"} {leg.ticker}:{" "}
                    {format(Math.abs(leg.weight) * 100)}% · {leg.role}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
      <div className="space-y-3 border-t border-slate-700 pt-4">
        <h4 className="font-semibold">
          Company fundamentals: supporting context
        </h4>
        <p className="text-xs text-slate-400">
          These are reported company figures, not inputs to this price-based
          signal. Reporting periods may differ, and filings can postdate the
          signal. They do not establish that the residual will converge.
        </p>
        {row.relative_value.references.length > 1 && (
          <label className="flex flex-wrap items-center gap-2">
            Compare {row.ticker_a} with
            <select
              className="rounded border border-slate-600 bg-[#0c141e] p-2"
              value={compare}
              onChange={(e) => setCompare(e.target.value)}
            >
              {row.relative_value.references.map((r) => (
                <option key={r.ticker}>{r.ticker}</option>
              ))}
            </select>
          </label>
        )}
        {!facts.length ? (
          <p className="text-slate-400">Loading reported fundamentals…</p>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {facts.map((company) => (
              <div
                key={company.ticker}
                className="min-w-0 rounded-lg border border-slate-700 p-3"
              >
                <b>
                  {company.ticker} · {company.name ?? "Company data"}
                </b>
                <p className="text-xs text-slate-400">
                  Source: {company.source}
                  {company.fetched_at
                    ? ` · fetched ${new Date(company.fetched_at).toLocaleString()}`
                    : ""}
                </p>
                {row.discovery?.profiles[company.ticker] && (
                  <p className="mt-1 text-xs text-slate-400">
                    {row.discovery.profiles[company.ticker].industry} ·{" "}
                    {row.discovery.profiles[company.ticker].category} (profile:{" "}
                    {row.discovery.profiles[company.ticker].source})
                  </p>
                )}
                <ul className="mt-2 space-y-2">
                  {company.metrics.slice(0, 6).map((metric) => (
                    <li key={metric.label}>
                      <div className="flex flex-wrap justify-between gap-2">
                        <span>{metric.label}</span>
                        <b>
                          {metric.value === null
                            ? "Unavailable"
                            : `${format(metric.value)} ${metric.unit}`}
                        </b>
                      </div>
                      <p className="text-xs text-slate-400">
                        {metric.basis} · period ending{" "}
                        {metric.period_end ?? "unavailable"} · filed{" "}
                        {metric.filed ?? "unavailable"}{" "}
                        {metric.filing_url &&
                          /^https?:\/\//i.test(metric.filing_url) && (
                            <a
                              className="text-emerald-300 underline"
                              href={metric.filing_url}
                              target="_blank"
                              rel="noreferrer"
                            >
                              Source filing
                            </a>
                          )}
                      </p>
                    </li>
                  ))}
                </ul>
                {!company.metrics.length && (
                  <p className="mt-2 text-amber-300">
                    Fundamentals unavailable for this security.
                  </p>
                )}
                {company.notes.map((note, i) => (
                  <p
                    key={i}
                    className="mt-2 break-words text-xs text-slate-400"
                  >
                    {note}
                  </p>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="space-y-2 border-t border-slate-700 pt-4">
        <h4 className="font-semibold">Check historical performance</h4>
        <p className="text-xs text-slate-400">
          Runs the same C++ model over the year ending at the signal date, with
          $100,000 initial cash, $1 per fill, 5 bps slippage and 3% annual short
          borrow. Includes every hedge leg and compares the same evaluated
          period with SPY. Change assumptions in Research.
        </p>
        <button
          className={button}
          disabled={
            busy ||
            !snapshot ||
            (!!run && ["queued", "running"].includes(run.status))
          }
          onClick={async () => {
            if (!snapshot) return;
            setBusy(true);
            setError("");
            try {
              const draft = await api<Record<string, unknown>>(
                "pairs/relative-value/strategy",
                {
                  method: "POST",
                  body: JSON.stringify({ relative_value: row.relative_value }),
                },
              );
              const end = snapshot.as_of;
              const start = new Date(
                new Date(end + "T00:00:00Z").getTime() - 365 * 86400000,
              )
                .toISOString()
                .slice(0, 10);
              setRun(
                await api<Run>("backtests", {
                  method: "POST",
                  body: JSON.stringify({
                    ...draft,
                    ticker: row.ticker_a,
                    interval: "1d",
                    start_date: start,
                    end_date: end,
                    initial_cash: 100000,
                    commission: 1,
                    slippage_bps: 5,
                    borrow_rate_percent: 3,
                    max_gross_exposure: 1,
                  }),
                }),
              );
            } catch (e) {
              setError(e instanceof Error ? e.message : "Backtest failed.");
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? "Submitting…" : "Backtest this setup"}
        </button>{" "}
        <Link className="text-emerald-300 underline" href="/research">
          Open Research & saved runs
        </Link>
        {error && <p className="text-amber-300">{error}</p>}
        {run && (
          <p role="status">
            Backtest: {run.status}
            {run.error ? ` · ${run.error}` : ""}
          </p>
        )}
        {run?.result && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr>
                  <th>Portfolio</th>
                  <th>Return</th>
                  <th>Max drawdown</th>
                  <th>Volatility</th>
                  <th>SPY beta</th>
                </tr>
              </thead>
              <tbody>
                {["strategy", "spy"].map((key) => {
                  const r = run.result!.risk_comparison[key];
                  return (
                    <tr key={key}>
                      <td className="py-2">
                        {key === "spy" ? "SPY buy & hold" : "Strategy"}
                      </td>
                      <td>{r.total_return_percent.toFixed(2)}%</td>
                      <td>{r.max_drawdown_percent.toFixed(2)}%</td>
                      <td>{r.volatility_percent.toFixed(2)}%</td>
                      <td>{r.beta_to_spy?.toFixed(3) ?? "Unavailable"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="text-xs text-slate-400">
              {run.result.strategy_diagnostics && <span>Backtest cost filter rejected {run.result.strategy_diagnostics.entries_rejected_by_cost} entries; skipped {run.result.strategy_diagnostics.rebalances_skipped} small or uneconomic rebalances. </span>}
              {run.result.fills.length} fills including hedge legs. Strategy
              invested{" "}
              {run.result.risk_comparison.strategy.time_in_market_percent.toFixed(
                1,
              )}
              % of evaluated observations.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
