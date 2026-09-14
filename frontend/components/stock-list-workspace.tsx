"use client";
import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/stockassist";
import {
  familyName,
  ListPlan,
  Opportunity,
  StockList,
} from "@/lib/stock-lists";
import OpportunityDetails from "./opportunity-details";
import PeerValuation from "./peer-valuation";

const field =
  "w-full min-w-0 rounded-lg border border-slate-700 bg-[#0c141e] p-2 text-sm";
const button =
  "rounded-lg border border-slate-600 px-3 py-2 text-sm disabled:opacity-40";
const primary =
  "rounded-lg bg-emerald-400 px-4 py-2 text-sm font-semibold text-emerald-950 disabled:opacity-40";
const algorithms = [
  ["residual_pair", "Residual pairs"],
  ["basket", "Stock versus basket"],
  ["peer_valuation", "Peer fundamental valuation (11+ eligible companies)"],
  ["same_company", "Known share classes"],
  ["etf", "Known ETF tracking pairs"],
  ["lead_lag", "Daily lead-lag (experimental)"],
];
const defaults = {
  algorithms: ["residual_pair", "basket", "same_company", "etf"],
  interval_hours: 24,
  max_alerts: 10,
  lookback: 120,
  threshold: 2,
  valuation_min_gap: 0.2,
  valuation_peers: [] as string[],
  min_correlation: 0.6,
  sector: "auto",
  market: "SPY",
  factors: ["MTUM", "VLUE"],
  basket_size: 5,
  model_limit: 20,
};
function ActionCard({
  row,
  onChange,
}: {
  row: Opportunity;
  onChange: () => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const entry = row.action.state === "entry";
  const change = async () => {
    setBusy(true);
    setError("");
    try {
      await api(`pairs/alerts/${row.id}/state`, {
        method: "POST",
        body: JSON.stringify({ active: !row.active }),
      });
      await onChange();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not update monitor.");
    } finally {
      setBusy(false);
    }
  };
  return (
    <article
      aria-label={`${row.ticker_a} ${familyName[row.relative_value.category] ?? row.relative_value.category} result`}
      className={`min-w-0 space-y-3 rounded-xl border p-4 ${entry ? "border-emerald-400/60 bg-emerald-400/5" : "border-slate-700 bg-[#101923]"}`}
    >
      <div className="flex flex-wrap justify-between gap-2">
        <div>
          <h3 className="font-semibold">
            {row.ticker_a} /{" "}
            {row.relative_value.references.map((r) => r.ticker).join(", ")}
          </h3>
          <p className="text-xs text-slate-400">
            {familyName[row.relative_value.category] ??
              row.relative_value.category}{" "}
            · {row.latest?.source ?? "Awaiting data"} ·{" "}
            {row.latest?.as_of ?? "No observation"}
          </p>
        </div>
        <b className={entry ? "text-emerald-300" : "text-slate-300"}>
          {row.action.label}
        </b>
      </div>
      {entry ? (
        <>
          <div className="grid gap-3 sm:grid-cols-2">
            {["long", "short"].map((side) => (
              <div
                key={side}
                className="rounded-lg border border-slate-700 p-3"
              >
                <h4
                  className={`text-sm font-bold ${side === "long" ? "text-emerald-300" : "text-rose-300"}`}
                >
                  {side === "long" ? "BUY / LONG" : "SELL SHORT"}
                </h4>
                <ul className="mt-2 space-y-1 text-sm">
                  {row.action.legs
                    .filter((leg) => leg.side === side)
                    .map((leg) => (
                      <li
                        key={leg.ticker}
                        className="flex flex-wrap justify-between gap-2"
                      >
                        <b>{leg.ticker}</b>
                        <span>
                          {(Math.abs(leg.weight) * 100).toFixed(2)}% ·{" "}
                          {leg.role}
                        </span>
                      </li>
                    ))}
                </ul>
                {!row.action.legs.some((leg) => leg.side === side) && (
                  <p className="text-sm text-slate-400">
                    No legs on this side.
                  </p>
                )}
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-400">
            Percentages are shares of the total gross allocation, not share
            quantities. Include every listed hedge leg. Review the backtest,
            current prices and short availability before deciding to place
            orders. No order has been placed.
          </p>
        </>
      ) : (
        <p className="text-sm text-slate-400">
          {row.action.state === "wait"
            ? `Nothing to buy or short from this setup now. Residual score ${row.latest?.value.toFixed(2)}; entry magnitude ${row.action.entry_z}.`
            : (row.last_error ??
              "Wait for a valid, active signal before considering an entry.")}
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        <button
          className={button}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {open
            ? "Hide explanation"
            : "Why this signal? Fundamentals & backtest"}
        </button>
        <button disabled={busy} className={button} onClick={change}>
          {row.active ? "Pause this monitor" : "Resume this monitor"}
        </button>
      </div>
      {error && <p className="text-sm text-amber-300">{error}</p>}
      {open && <OpportunityDetails row={row} />}
    </article>
  );
}
export default function StockListWorkspace() {
  const [lists, setLists] = useState<StockList[]>([]);
  const [plans, setPlans] = useState<ListPlan[]>([]);
  const [rows, setRows] = useState<Opportunity[]>([]);
  const [selected, setSelected] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  const [formOpen, setFormOpen] = useState(false);
  const [name, setName] = useState("");
  const [symbols, setSymbols] = useState("");
  const [settings, setSettings] = useState(defaults);
  const [valuationPeers, setValuationPeers] = useState("");
  const [factorText, setFactorText] = useState("MTUM, VLUE");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [errors, setErrors] = useState("");
  const [feedError, setFeedError] = useState("");
  const [onlyEntries, setOnlyEntries] = useState(false);
  const refresh = useCallback(async () => {
    const [l, p, r] = await Promise.all([
      api<StockList[]>("pairs/stock-lists"),
      api<ListPlan[]>("pairs/schedules"),
      api<Opportunity[]>("pairs/opportunities"),
    ]);
    setLists(l);
    setPlans(p.filter((v) => v.config.source === "stock_list"));
    setRows(r);
    setFeedError("");
  }, []);
  useEffect(() => {
    let alive = true;
    const load = () => {
      void refresh().catch((e) => {
        if (alive) setFeedError(e.message);
      });
    };
    load();
    const timer = setInterval(load, 5000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [refresh]);
  const list = lists.find((l) => l.id === selected);
  const plan = plans.find((p) => p.config.stock_list_id === selected);
  const availableRows = feedError
    ? rows.map((row) => ({
        ...row,
        last_error: feedError,
        action: {
          ...row.action,
          state: "error",
          label: "Unable to refresh — wait for data",
          legs: [],
        },
      }))
    : rows;
  const matching = availableRows
    .filter(
      (row) =>
        !plan ||
        plan.config.algorithms.includes(
          row.relative_value.category === "competitor"
            ? "residual_pair"
            : row.relative_value.category,
        ),
    )
    .filter(
      (row) =>
        !list ||
        (list.tickers.includes(row.ticker_a) &&
          row.relative_value.references.every((r) =>
            list.tickers.includes(r.ticker),
          )),
    );
  const visible = matching
    .filter((row) => !onlyEntries || row.action.state === "entry")
    .sort(
      (a, b) =>
        Number(b.action.state === "entry") -
          Number(a.action.state === "entry") ||
        Math.abs(b.latest?.value ?? 0) - Math.abs(a.latest?.value ?? 0),
    );
  async function mutate(fn: () => Promise<void>) {
    setBusy(true);
    setErrors("");
    setMessage("");
    try {
      await fn();
      await refresh();
    } catch (e) {
      setErrors(e instanceof Error ? e.message : "Request failed.");
      await refresh().catch(() => {});
    } finally {
      setBusy(false);
    }
  }
  function editList(value: StockList) {
    const p = plans.find((row) => row.config.stock_list_id === value.id);
    setEditing(value.id);
    setName(value.name);
    setSymbols(value.tickers.join(", "));
    setSettings(p ? { ...defaults, ...p.config } : defaults);
    setValuationPeers((p?.config.valuation_peers ?? []).join(", "));
    setFactorText((p?.config.factors ?? defaults.factors).join(", "));
    setFormOpen(true);
  }
  return (
    <section aria-label="Automated stock list strategies" className="space-y-6">
      <div className="rounded-2xl border border-[#243141] bg-[#101923] p-6">
        <h2 className="text-lg font-semibold">
          1. Choose your stocks. We check the strategies.
        </h2>
        <p className="mt-2 text-sm text-slate-400">
          Save a list of 2–20 US stocks or ETFs. The C++ backend finds eligible
          relationships, evaluates the selected models on a schedule, and keeps
          valid setups monitored. Keep the backend and TWS/IB Gateway running.
        </p>
        <div className="mt-4 flex flex-wrap items-end gap-3">
          <label className="min-w-0 flex-1 text-sm">
            Stock list
            <select
              aria-label="Stock list"
              className={`${field} mt-2`}
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
            >
              <option value="">All existing monitors</option>
              {lists.map((l) => (
                <option key={l.id} value={l.id}>
                  {l.name} ({l.tickers.length} stocks)
                </option>
              ))}
            </select>
          </label>
          <button
            className={button}
            onClick={() => {
              setEditing(null);
              setName("");
              setSymbols("");
              setSettings(defaults);
              setValuationPeers("");
              setFactorText(defaults.factors.join(", "));
              setFormOpen(true);
            }}
          >
            Create stock list
          </button>
          {list && (
            <>
              <button className={button} onClick={() => editList(list)}>
                Edit list & strategies
              </button>
              <button
                className={button}
                disabled={busy}
                onClick={() =>
                  mutate(async () => {
                    await api(`pairs/stock-lists/${list.id}`, {
                      method: "DELETE",
                    });
                    setSelected("");
                    setFormOpen(false);
                    setMessage(
                      "List deleted. Its scans stopped and monitors it created were paused; event history remains.",
                    );
                  })
                }
              >
                Delete list
              </button>
            </>
          )}
        </div>
        {(formOpen || !lists.length) && (
          <form
            className="mt-5 space-y-4 border-t border-slate-700 pt-5"
            onSubmit={(e) => {
              e.preventDefault();
              void mutate(async () => {
                const saved = await api<StockList>(
                  `pairs/stock-lists${editing ? `/${editing}` : ""}`,
                  {
                    method: editing ? "PUT" : "POST",
                    body: JSON.stringify({
                      name,
                      tickers: symbols.split(/[\s,]+/).filter(Boolean),
                    }),
                  },
                );
                setSelected(saved.id);
                setEditing(saved.id);
                const existing = plans.find(
                  (p) => p.config.stock_list_id === saved.id,
                );
                await api(
                  `pairs/schedules${existing ? `/${existing.id}` : ""}`,
                  {
                    method: existing ? "PUT" : "POST",
                    body: JSON.stringify({
                      ...settings,
                      valuation_peers: valuationPeers
                        .split(/[\s,]+/)
                        .filter(Boolean),
                      factors: factorText.split(/[\s,]+/).filter(Boolean),
                      source: "stock_list",
                      stock_list_id: saved.id,
                    }),
                  },
                );
                setFormOpen(false);
                setEditing(null);
                setMessage(
                  existing && !existing.active
                    ? "List saved. Its schedule remains paused; resume it below to run."
                    : "List saved. The first scan starts shortly; valid setups will appear below automatically.",
                );
              });
            }}
          >
            <div className="grid gap-4 sm:grid-cols-2">
              <label className="grid gap-2 text-sm">
                List name
                <input
                  className={field}
                  required
                  maxLength={100}
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Name your stock list"
                />
              </label>
              <label className="grid gap-2 text-sm">
                Stocks to compare
                <input
                  className={field}
                  required
                  value={symbols}
                  onChange={(e) => setSymbols(e.target.value)}
                  placeholder="Enter 2–20 symbols, separated by commas"
                />
              </label>
            </div>
            <fieldset>
              <legend className="mb-2 text-sm font-semibold">
                Strategies to check automatically
              </legend>
              <div className="flex flex-wrap gap-4">
                {algorithms.map(([key, label]) => (
                  <label key={key} className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={settings.algorithms.includes(key)}
                      onChange={(e) =>
                        setSettings((s) => ({
                          ...s,
                          algorithms: e.target.checked
                            ? [...s.algorithms, key]
                            : s.algorithms.filter((a) => a !== key),
                        }))
                      }
                    />
                    {label}
                  </label>
                ))}
              </div>
            </fieldset>
            <p className="text-xs text-slate-400">
              Residual pairs and baskets use matching provider industry/category
              profiles, positive correlation and inferred sector hedges. Unknown
              profiles are reported for review. Share-class/ETF checks use
              documented relationships supported by the app; arbitrary list
              membership is not proof of equivalence. Event terms cannot be
              inferred from a stock list.
            </p>
            {settings.algorithms.includes("peer_valuation") && (
              <label className="grid gap-2 text-sm">
                Additional valuation reference peers (optional)
                <textarea
                  className={field}
                  value={valuationPeers}
                  onChange={(e) => setValuationPeers(e.target.value)}
                  placeholder="Up to 40 economically comparable stock symbols"
                />
                <span className="text-xs text-slate-400">
                  Reference peers help estimate valuations; only stocks in your
                  main list are candidates or basket references. At least 10
                  other eligible issuers per target are required. Choose a
                  sector ETF in the optional settings for price confirmation.
                </span>
              </label>
            )}
            <details className="rounded-lg border border-slate-700 p-3">
              <summary className="cursor-pointer text-sm">
                Optional scan and risk settings
              </summary>
              <div className="mt-3 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {(
                  [
                    ["interval_hours", "Check list every (hours)", 1, 168, 1],
                    ["max_alerts", "Maximum generated monitors", 1, 50, 1],
                    ["lookback", "Estimation days", 40, 250, 1],
                    ["threshold", "Entry score magnitude", 0.1, 20, 0.1],
                    [
                      "valuation_min_gap",
                      "Minimum fundamental valuation gap (fraction)",
                      0.05,
                      1,
                      0.05,
                    ],
                    ["min_correlation", "List correlation filter", 0, 1, 0.05],
                    ["basket_size", "Maximum basket constituents", 2, 10, 1],
                    ["model_limit", "Models evaluated per scan", 1, 30, 1],
                  ] as const
                ).map(([key, label, min, max, step]) => (
                  <label key={key} className="grid gap-2 text-sm">
                    {label}
                    <input
                      className={field}
                      type="number"
                      required
                      min={min}
                      max={max}
                      step={step}
                      value={settings[key]}
                      onChange={(e) =>
                        setSettings((s) => ({
                          ...s,
                          [key]: Number(e.target.value),
                        }))
                      }
                    />
                  </label>
                ))}
                <label className="grid gap-2 text-sm">
                  Sector hedge selection
                  <select
                    className={field}
                    value={settings.sector}
                    onChange={(e) =>
                      setSettings((s) => ({ ...s, sector: e.target.value }))
                    }
                  >
                    <option value="auto">
                      Automatic from provider classification
                    </option>
                    {[
                      ["XLK", "Technology"],
                      ["XLF", "Financials"],
                      ["XLE", "Energy"],
                      ["XLV", "Health care"],
                      ["XLC", "Communication"],
                      ["XLI", "Industrials"],
                      ["XLP", "Consumer staples"],
                      ["XLY", "Consumer discretionary"],
                      ["XLB", "Materials"],
                      ["XLRE", "Real estate"],
                      ["XLU", "Utilities"],
                    ].map(([symbol, label]) => (
                      <option key={symbol} value={symbol}>
                        {label} — {symbol} (override for whole list)
                      </option>
                    ))}
                  </select>
                </label>
                <label className="grid gap-2 text-sm">
                  Market hedge
                  <input
                    className={field}
                    value={settings.market}
                    onChange={(e) =>
                      setSettings((s) => ({ ...s, market: e.target.value }))
                    }
                  />
                </label>
                <label className="grid gap-2 text-sm">
                  Factor hedge ETFs
                  <input
                    className={field}
                    value={factorText}
                    onChange={(e) => setFactorText(e.target.value)}
                  />
                </label>
              </div>
              <p className="mt-3 text-xs text-slate-400">
                Overrides declare your own common sector; the app does not
                verify that every stock belongs to it. A model cannot use the
                same ticker as a traded leg and a hedge. Existing monitors
                retain their model settings; removing stocks pauses affected
                monitors created by this list.
              </p>
            </details>
            <div className="flex flex-wrap gap-2">
              <button
                className={primary}
                disabled={busy || !settings.algorithms.length}
              >
                {busy
                  ? "Saving…"
                  : editing
                    ? "Save list & scan settings"
                    : "Save list & start automatic checks"}
              </button>
              {lists.length > 0 && (
                <button
                  type="button"
                  className={button}
                  onClick={() => setFormOpen(false)}
                >
                  Cancel
                </button>
              )}
            </div>
          </form>
        )}
        {list && (
          <div className="mt-4 space-y-2 rounded-lg border border-slate-700 p-3 text-sm">
            <p>
              <b>{list.name}</b>: {list.tickers.join(", ")}
            </p>
            {plan ? (
              <>
                <p>
                  {plan.running
                    ? "Scanning your list…"
                    : plan.active
                      ? "Automatic checks enabled"
                      : "Discovery paused"}{" "}
                  · every {plan.config.interval_hours}h · Next:{" "}
                  {plan.active
                    ? new Date(plan.next_run_at).toLocaleString()
                    : "paused"}
                </p>
                {plan.last_result && (
                  <p>
                    {plan.last_result.evaluated ?? plan.last_result.candidates}{" "}
                    models evaluated · {plan.last_result.candidates} valid ·{" "}
                    {plan.last_result.created.length} monitors added last run ·{" "}
                    {plan.last_result.managed_alerts}/{plan.config.max_alerts}{" "}
                    generated monitors at last scan
                  </p>
                )}
                <div className="flex flex-wrap gap-2">
                  <button
                    className={button}
                    disabled={busy || plan.running || !plan.active}
                    onClick={() =>
                      mutate(async () => {
                        await api(`pairs/schedules/${plan.id}/run`, {
                          method: "POST",
                        });
                        setMessage(
                          "Scan queued. Results update automatically.",
                        );
                      })
                    }
                  >
                    Check my list now
                  </button>
                  <button
                    className={button}
                    disabled={busy}
                    onClick={() =>
                      mutate(async () => {
                        await api(`pairs/schedules/${plan.id}/state`, {
                          method: "POST",
                          body: JSON.stringify({ active: !plan.active }),
                        });
                      })
                    }
                  >
                    {plan.active
                      ? "Pause list discovery"
                      : "Resume list discovery"}
                  </button>
                </div>
                <p className="text-xs text-slate-400">
                  Pausing discovery leaves existing monitors active. Pause
                  individual monitors below to stop their signals. Monitor
                  limits include paused rules.
                </p>
                {(plan.last_error ||
                  !!plan.last_result?.errors.length ||
                  !!plan.last_result?.skipped?.length) && (
                  <details className="rounded border border-amber-400/30 p-2">
                    <summary className="cursor-pointer text-amber-300">
                      Data issues or relationships skipped — review details
                    </summary>
                    {plan.last_error && (
                      <p className="mt-2 break-words text-amber-300">
                        {plan.last_error}
                      </p>
                    )}
                    {plan.last_result?.errors.map((e, i) => (
                      <p key={i} className="mt-1 text-xs">
                        {e.ticker}: {e.error}
                      </p>
                    ))}
                    {plan.last_result?.skipped?.map((e, i) => (
                      <p key={i} className="mt-1 text-xs">
                        {e.ticker}: {e.reason}
                      </p>
                    ))}
                  </details>
                )}
              </>
            ) : (
              <>
                <p>No strategy schedule is saved for this list yet.</p>
                <button className={button} onClick={() => editList(list)}>
                  Choose strategies & enable checks
                </button>
              </>
            )}
          </div>
        )}
        {message && (
          <p role="status" className="mt-3 text-sm text-emerald-300">
            {message}
          </p>
        )}
        {feedError && (
          <p role="alert" className="mt-3 text-sm text-amber-300">
            {feedError}
          </p>
        )}
        {errors && (
          <p role="alert" className="mt-3 break-words text-sm text-amber-300">
            {errors}
          </p>
        )}
      </div>
      {plan?.config.algorithms.includes("peer_valuation") && (
        <PeerValuation
          result={plan.last_result?.valuation}
          unavailable={
            !!feedError ||
            !plan.active ||
            !!plan.last_error ||
            (!!plan.last_result?.valuation &&
              Date.now() - Date.parse(plan.last_result.valuation.as_of) >
                5 * 86400000)
          }
        />
      )}
      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold">2. Review what to do</h2>
            <p className="mt-1 text-sm text-slate-400">
              {matching.filter((r) => r.action.state === "entry").length} entry
              candidates · {matching.length} matching monitors. Open an
              explanation for the model, fundamentals and a one-click backtest.
            </p>
          </div>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={onlyEntries}
              onChange={(e) => setOnlyEntries(e.target.checked)}
            />
            Show entry candidates only
          </label>
        </div>
        {!visible.length && (
          <div className="rounded-xl border border-slate-700 p-5 text-sm text-slate-400">
            {onlyEntries
              ? "No entry candidates. There is nothing to buy or short from these signals now."
              : plan?.running
                ? "Your list is being analyzed. Valid setups will appear here automatically."
                : "No matching monitors yet. Save a list and enable its strategy checks, or review any scan issues above."}
          </div>
        )}
        <div className="space-y-4">
          {visible.map((row) => (
            <ActionCard key={row.id} row={row} onChange={refresh} />
          ))}
        </div>
      </div>
    </section>
  );
}
