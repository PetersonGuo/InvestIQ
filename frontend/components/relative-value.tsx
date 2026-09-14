"use client";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/stockassist";

type Configuration = {
  category: string;
  target: string;
  references: { ticker: string; weight: number }[];
  market: string;
  sector: string;
  factors: string[];
  fit_reference: boolean;
  lookback: number;
  entry_z: number;
  cash_per_share?: number;
  shares_per_share?: number;
  terms_as_of?: string;
  expires_on?: string;
  terms_source?: string;
};
type Catalog = {
  id: string;
  name: string;
  description: string;
  config: Configuration;
};
type Snapshot = {
  value: number;
  matched: boolean;
  as_of: string;
  source: string;
  baseline_start: string;
  baseline_end: string;
  signal: { direction_text: string } | null;
  relative_value: {
    score_type: string;
    fit_r2: number;
    event_gap_percent: number | null;
    neutrality: string;
    hedge_legs: {
      ticker: string;
      side: string;
      weight: number;
      role: string;
    }[];
    exposures: {
      ticker: string;
      role: string;
      before_beta: number;
      after_beta: number;
    }[];
  };
};
type Alert = {
  id: string;
  relative_value?: Configuration;
  active: number;
  latest: Snapshot | null;
  last_error: string | null;
};
type Schedule = {
  id: string;
  config: {
    source: string;
    interval_hours: number;
    max_alerts: number;
    relationships: { relative_value: Configuration }[];
  };
  active: boolean;
  running: boolean;
  next_run_at: string;
  last_error: string | null;
  last_result: {
    candidates: number;
    created: string[];
    errors: { ticker: string; error: string }[];
  } | null;
};
type Event = {
  id: string;
  created_at: string;
  config: { relative_value?: Configuration };
  snapshot: Snapshot;
};
const input =
  "w-full min-w-0 rounded-lg border border-slate-700 bg-[#0c141e] p-2 text-sm";
const button =
  "rounded-lg border border-slate-600 px-3 py-2 text-sm disabled:opacity-40";
const primary =
  "rounded-lg bg-emerald-400 px-4 py-2 text-sm font-semibold text-emerald-950 disabled:opacity-40";

function Diagnostics({ snapshot }: { snapshot: Snapshot }) {
  const model = snapshot.relative_value;
  return (
    <div className="space-y-3 text-sm">
      <p>
        <b>
          {model.score_type}: {snapshot.value.toFixed(3)}
        </b>{" "}
        · {snapshot.source} · {snapshot.as_of}
      </p>
      <p className="text-emerald-300">
        {snapshot.signal?.direction_text ??
          "No entry signal at the configured threshold."}
      </p>
      <p className="text-slate-400">
        Fit window: {snapshot.baseline_start}–{snapshot.baseline_end} ·
        In-sample R²: {model.fit_r2.toFixed(3)}
      </p>
      {model.event_gap_percent !== null && (
        <p>
          Contractual payoff gap: {model.event_gap_percent.toFixed(3)}% (target
          / cash-plus-stock value − 1)
        </p>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <caption className="pb-2 text-left font-semibold">
            Estimated exposure before / after hedging
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
            {model.exposures.map((row) => (
              <tr key={row.ticker} className="border-t border-slate-800">
                <td className="p-2">{row.ticker}</td>
                <td>{row.role}</td>
                <td>{row.before_beta.toFixed(4)}</td>
                <td>{row.after_beta.toFixed(6)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-slate-400">{model.neutrality}</p>
      <div className="flex flex-wrap gap-2">
        {model.hedge_legs.map((leg) => (
          <span
            key={leg.ticker}
            className="rounded border border-slate-700 px-2 py-1 text-xs"
          >
            {leg.side} {leg.ticker} · {(100 * Math.abs(leg.weight)).toFixed(2)}%
            gross · {leg.role}
          </span>
        ))}
      </div>
      <p className="text-xs text-slate-400">
        Weights show the portfolio for the current signal direction, including
        all exposure hedges. Below the entry threshold these are analytical
        weights, not an entry alert. Return residuals are not a cointegration
        test. Whole shares, costs and price changes can leave realized exposure.
      </p>
    </div>
  );
}

export default function RelativeValue() {
  const [catalog, setCatalog] = useState<Catalog[]>([]);
  const [config, setConfig] = useState<Configuration | null>(null);
  const [references, setReferences] = useState("");
  const [factors, setFactors] = useState("");
  const [preview, setPreview] = useState<Snapshot | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  const [searchList, setSearchList] = useState<Configuration[]>([]);
  const [hours, setHours] = useState(24);
  const [cap, setCap] = useState(5);
  const [editingSchedule, setEditingSchedule] = useState<string | null>(null);
  const [editingAlert, setEditingAlert] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const refresh = useCallback(async () => {
    const [rules, jobs, notifications] = await Promise.all([
      api<Alert[]>("pairs/alerts"),
      api<Schedule[]>("pairs/schedules"),
      api<Event[]>("pairs/events"),
    ]);
    setAlerts(rules.filter((row) => row.relative_value));
    setSchedules(jobs.filter((row) => row.config.source === "relationships"));
    setEvents(notifications.filter((row) => row.config.relative_value));
  }, []);
  function load(value: Configuration) {
    setConfig(value);
    setReferences(
      value.references.map((r) => `${r.ticker}:${r.weight}`).join(", "),
    );
    setFactors(value.factors.join(", "));
    setPreview(null);
  }
  useEffect(() => {
    void api<Catalog[]>("pairs/relative-value/catalog")
      .then((rows) => {
        setCatalog(rows);
        if (rows[2]) load(rows[2].config);
      })
      .catch((e) => setMessage(e.message));
    const update = () => {
      void refresh().catch((e) => setMessage(e.message));
    };
    update();
    const timer = setInterval(update, 10000);
    return () => clearInterval(timer);
  }, [refresh]);
  function change<K extends keyof Configuration>(
    key: K,
    value: Configuration[K],
  ) {
    setConfig((c) => (c ? { ...c, [key]: value } : c));
    setPreview(null);
  }
  function payload(): Configuration {
    if (!config) throw new Error("Select a relationship.");
    return {
      ...config,
      references: references
        .split(/[\s,]+/)
        .filter(Boolean)
        .map((s) => {
          const [ticker, weight] = s.split(":");
          const n = weight === undefined ? 1 : Number(weight);
          if (!Number.isFinite(n))
            throw new Error("Reference weights must be numbers.");
          return { ticker: ticker.toUpperCase(), weight: n };
        }),
      factors: factors
        .split(/[\s,]+/)
        .filter(Boolean)
        .map((s) => s.toUpperCase()),
    };
  }
  async function action(fn: () => Promise<void>) {
    setBusy(true);
    setMessage("");
    try {
      await fn();
      await refresh();
    } catch (e) {
      setMessage(
        e instanceof Error ? e.message : "Relative-value request failed.",
      );
    } finally {
      setBusy(false);
    }
  }
  const description = catalog.find(
    (row) => row.id === config?.category,
  )?.description;
  return (
    <section
      aria-label="Exposure-adjusted relative value"
      className="space-y-5 rounded-2xl border border-[#243141] bg-[#101923] p-6"
    >
      <div>
        <h2 className="font-semibold">Exposure-adjusted relative value</h2>
        <p className="mt-2 text-sm text-slate-400">
          Keep the security-specific residual; hedge estimated market, sector
          and selected factor exposure. All calculations and exported strategies
          run in C++.{" "}
          <Link
            className="text-emerald-300 underline"
            href="/docs/api#relative-value-models"
          >
            Model and API guide
          </Link>
        </p>
      </div>
      {config && (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <label className="grid gap-2 text-sm">
              Relationship family
              <select
                className={input}
                value={config.category}
                onChange={(e) => {
                  const item = catalog.find((row) => row.id === e.target.value);
                  if (item) {
                    load(item.config);
                    setEditingAlert(null);
                  }
                }}
              >
                {catalog.map((row) => (
                  <option key={row.id} value={row.id}>
                    {row.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="grid gap-2 text-sm">
              Target security
              <input
                className={input}
                value={config.target}
                onChange={(e) => change("target", e.target.value)}
              />
            </label>
            <label className="grid gap-2 text-sm">
              Reference basket (ticker:weight)
              <input
                className={input}
                value={references}
                onChange={(e) => {
                  setReferences(e.target.value);
                  setPreview(null);
                }}
              />
            </label>
            <label className="grid gap-2 text-sm">
              Market hedge ETF
              <input
                className={input}
                value={config.market}
                onChange={(e) => change("market", e.target.value)}
              />
            </label>
            <label className="grid gap-2 text-sm">
              Sector / broad exposure hedge ETF
              <input
                className={input}
                value={config.sector}
                onChange={(e) => change("sector", e.target.value)}
              />
            </label>
            <label className="grid gap-2 text-sm">
              Additional factor ETF proxies
              <input
                className={input}
                value={factors}
                onChange={(e) => {
                  setFactors(e.target.value);
                  setPreview(null);
                }}
              />
            </label>
            <label className="grid gap-2 text-sm">
              Residual estimation days
              <input
                className={input}
                type="number"
                min={40}
                max={250}
                value={config.lookback}
                onChange={(e) => change("lookback", Number(e.target.value))}
              />
            </label>
            <label className="grid gap-2 text-sm">
              Residual entry threshold
              <input
                className={input}
                type="number"
                min={0.1}
                max={20}
                step={0.1}
                value={config.entry_z}
                onChange={(e) => change("entry_z", Number(e.target.value))}
              />
            </label>
            {!["lead_lag", "event"].includes(config.category) && (
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={config.fit_reference}
                  onChange={(e) => change("fit_reference", e.target.checked)}
                />
                Fit reference weights from prior returns
              </label>
            )}
            {config.category === "event" && (
              <>
                <label className="grid gap-2 text-sm">
                  Cash consideration per target share
                  <input
                    className={input}
                    type="number"
                    min={0}
                    value={config.cash_per_share}
                    onChange={(e) =>
                      change("cash_per_share", Number(e.target.value))
                    }
                  />
                </label>
                <label className="grid gap-2 text-sm">
                  Reference shares per target share
                  <input
                    className={input}
                    type="number"
                    min={0}
                    step="any"
                    value={config.shares_per_share}
                    onChange={(e) =>
                      change("shares_per_share", Number(e.target.value))
                    }
                  />
                </label>
                <label className="grid gap-2 text-sm">
                  Event terms known on
                  <input
                    className={input}
                    type="date"
                    value={config.terms_as_of}
                    onChange={(e) => change("terms_as_of", e.target.value)}
                  />
                </label>
                <label className="grid gap-2 text-sm">
                  Event terms valid through
                  <input
                    className={input}
                    type="date"
                    value={config.expires_on}
                    onChange={(e) => change("expires_on", e.target.value)}
                  />
                </label>
                <label className="grid gap-2 text-sm">
                  Event terms source
                  <input
                    className={input}
                    value={config.terms_source}
                    onChange={(e) => change("terms_source", e.target.value)}
                    placeholder="Filing URL or source reference"
                  />
                </label>
              </>
            )}
          </div>
          <p className="text-sm text-slate-400">{description}</p>
          <p className="text-xs text-slate-400">
            Daily, completed bars only. Use 1–20 references and 1–6 factor
            proxies; all symbols must be distinct. Fixed weights are
            dollar-return coefficients, not ADR conversion ratios. Factor ETFs
            are observable proxies, not pure academic factors. The selected
            relationships are research starting points, not validated
            opportunities.
          </p>
          <div className="flex flex-wrap gap-2">
            <button
              className={primary}
              disabled={busy}
              onClick={() =>
                action(async () => {
                  setPreview(null);
                  setPreview(
                    await api<Snapshot>("pairs/preview", {
                      method: "POST",
                      body: JSON.stringify({ relative_value: payload() }),
                    }),
                  );
                })
              }
            >
              Analyze residual
            </button>
            <button
              className={button}
              disabled={busy || !preview}
              onClick={() =>
                action(async () => {
                  await api(
                    `pairs/alerts${editingAlert ? `/${editingAlert}` : ""}`,
                    {
                      method: editingAlert ? "PUT" : "POST",
                      body: JSON.stringify({ relative_value: payload() }),
                    },
                  );
                  setEditingAlert(null);
                  setMessage(
                    "Persistent residual alert saved. All hedge legs are shown when it triggers.",
                  );
                })
              }
            >
              {editingAlert ? "Save residual alert" : "Watch residual"}
            </button>
            <button
              className={button}
              disabled={busy}
              onClick={() =>
                action(async () => {
                  const draft = await api<Record<string, unknown>>(
                    "pairs/relative-value/strategy",
                    {
                      method: "POST",
                      body: JSON.stringify({ relative_value: payload() }),
                    },
                  );
                  await api("strategies", {
                    method: "POST",
                    body: JSON.stringify(draft),
                  });
                  setMessage(
                    "C++ strategy saved in Research. Load it from saved strategies to choose dates and run against SPY.",
                  );
                })
              }
            >
              Save C++ backtest strategy
            </button>
            <button
              className={button}
              disabled={busy}
              onClick={() =>
                action(async () => {
                  const value = payload();
                  await api("pairs/relative-value/strategy", {
                    method: "POST",
                    body: JSON.stringify({ relative_value: value }),
                  });
                  setSearchList((list) => [...list, value]);
                  setMessage(
                    "Relationship added to the scheduled search list below.",
                  );
                })
              }
            >
              Add to scheduled search
            </button>
            {editingAlert && (
              <button className={button} onClick={() => setEditingAlert(null)}>
                Cancel alert edit
              </button>
            )}
          </div>
        </>
      )}
      <p role="status" className="break-words text-sm text-slate-300">
        {message}
      </p>
      {preview && <Diagnostics snapshot={preview} />}
      <div className="space-y-3 border-t border-slate-700 pt-4">
        <h3 className="font-semibold">Scheduled economic relationships</h3>
        <p className="text-sm text-slate-400">
          Build a list across families above. Each scheduled run evaluates these
          relationships and automatically creates persistent residual monitors
          when the data/model is valid. It does not infer economic equivalence
          from random ticker correlations. Existing monitors refit their hedges
          on new daily data; pausing discovery leaves them active.
        </p>
        <ul className="space-y-1">
          {searchList.map((v, i) => (
            <li
              key={i}
              className="flex flex-wrap items-center justify-between gap-2 text-sm"
            >
              <span>
                {v.category}: {v.target} /{" "}
                {v.references.map((r) => r.ticker).join(", ")}
              </span>
              <button
                className={button}
                onClick={() =>
                  setSearchList((list) => list.filter((_, j) => i !== j))
                }
              >
                Remove relationship {i + 1}
              </button>
            </li>
          ))}
        </ul>
        <div className="flex flex-wrap items-end gap-3">
          <label className="grid gap-2 text-sm">
            Residual scan interval (hours)
            <input
              className={input}
              type="number"
              min={1}
              max={168}
              value={hours}
              onChange={(e) => setHours(Number(e.target.value))}
            />
          </label>
          <label className="grid gap-2 text-sm">
            Residual monitor cap
            <input
              className={input}
              type="number"
              min={1}
              max={50}
              value={cap}
              onChange={(e) => setCap(Number(e.target.value))}
            />
          </label>
          <button
            disabled={busy || !searchList.length}
            className={primary}
            onClick={() =>
              action(async () => {
                await api(
                  `pairs/schedules${editingSchedule ? `/${editingSchedule}` : ""}`,
                  {
                    method: editingSchedule ? "PUT" : "POST",
                    body: JSON.stringify({
                      source: "relationships",
                      relationships: searchList.map((relative_value) => ({
                        relative_value,
                      })),
                      interval_hours: hours,
                      max_alerts: cap,
                    }),
                  },
                );
                setSearchList([]);
                setEditingSchedule(null);
                setMessage("Economic relationship schedule saved.");
              })
            }
          >
            {editingSchedule
              ? "Save relationship schedule"
              : "Enable relationship schedule"}
          </button>
          {editingSchedule && (
            <button
              className={button}
              onClick={() => {
                setEditingSchedule(null);
                setSearchList([]);
              }}
            >
              Cancel schedule edit
            </button>
          )}
        </div>
        {schedules.map((row) => (
          <div
            key={row.id}
            className="space-y-2 rounded-lg border border-slate-700 p-3 text-sm"
          >
            <b>
              {row.config.relationships.length} relationships · every{" "}
              {row.config.interval_hours}h ·{" "}
              {!row.active ? "Paused" : row.running ? "Scanning…" : "Scheduled"}
            </b>
            <p>
              Next run:{" "}
              {row.active
                ? new Date(row.next_run_at).toLocaleString()
                : "Paused"}
            </p>
            {row.last_result && (
              <p>
                {row.last_result.candidates} valid models ·{" "}
                {row.last_result.created.length} monitors added last run
              </p>
            )}
            {row.last_error && (
              <p className="break-words text-amber-300">{row.last_error}</p>
            )}
            {row.last_result?.errors.map((e, i) => (
              <p key={i} className="text-amber-300">
                {e.ticker}: {e.error}
              </p>
            ))}
            <div className="flex flex-wrap gap-2">
              <button
                className={button}
                disabled={busy || !row.active || row.running}
                onClick={() =>
                  action(async () => {
                    await api(`pairs/schedules/${row.id}/run`, {
                      method: "POST",
                    });
                  })
                }
              >
                Scan relationships now
              </button>
              <button
                className={button}
                disabled={busy}
                onClick={() =>
                  action(async () => {
                    await api(`pairs/schedules/${row.id}/state`, {
                      method: "POST",
                      body: JSON.stringify({ active: !row.active }),
                    });
                  })
                }
              >
                {row.active
                  ? "Pause relationship schedule"
                  : "Resume relationship schedule"}
              </button>
              <button
                className={button}
                onClick={() => {
                  setEditingSchedule(row.id);
                  setSearchList(
                    row.config.relationships.map((r) => r.relative_value),
                  );
                  setHours(row.config.interval_hours);
                  setCap(row.config.max_alerts);
                }}
              >
                Edit relationship schedule
              </button>
              <button
                className={button}
                disabled={busy}
                onClick={() =>
                  action(async () => {
                    await api(`pairs/schedules/${row.id}`, {
                      method: "DELETE",
                    });
                    if (editingSchedule === row.id) {
                      setEditingSchedule(null);
                      setSearchList([]);
                    }
                  })
                }
              >
                Delete relationship schedule
              </button>
            </div>
          </div>
        ))}
      </div>
      <div className="space-y-3 border-t border-slate-700 pt-4">
        <h3 className="font-semibold">Persistent residual monitors</h3>
        {!alerts.length && (
          <p className="text-sm text-slate-400">No residual monitors saved.</p>
        )}
        {alerts.map((row) => (
          <details
            key={row.id}
            className="rounded-lg border border-slate-700 p-3 text-sm"
          >
            <summary className="cursor-pointer">
              {row.relative_value?.category}: {row.relative_value?.target} ·{" "}
              {row.active ? "Monitoring" : "Paused"} ·{" "}
              {row.latest?.value.toFixed(3) ?? "No observation"}
            </summary>
            <div className="mt-3 space-y-3">
              {row.last_error && (
                <p className="text-amber-300">{row.last_error}</p>
              )}
              {row.latest && <Diagnostics snapshot={row.latest} />}
              <div className="flex flex-wrap gap-2">
                <button
                  className={button}
                  disabled={busy}
                  onClick={() =>
                    action(async () => {
                      await api(`pairs/alerts/${row.id}/state`, {
                        method: "POST",
                        body: JSON.stringify({ active: !row.active }),
                      });
                    })
                  }
                >
                  {row.active
                    ? "Pause residual monitor"
                    : "Resume residual monitor"}
                </button>
                <button
                  className={button}
                  onClick={() => {
                    if (row.relative_value) {
                      load(row.relative_value);
                      setEditingAlert(row.id);
                    }
                  }}
                >
                  Edit residual monitor
                </button>
                <button
                  className={button}
                  disabled={busy}
                  onClick={() =>
                    action(async () => {
                      await api(`pairs/alerts/${row.id}`, { method: "DELETE" });
                      if (editingAlert === row.id) setEditingAlert(null);
                    })
                  }
                >
                  Delete residual monitor
                </button>
              </div>
            </div>
          </details>
        ))}
      </div>
      <div className="space-y-2 border-t border-slate-700 pt-4">
        <h3 className="font-semibold">Residual notifications</h3>
        {!events.length && (
          <p className="text-sm text-slate-400">
            No residual entry events recorded.
          </p>
        )}
        {events.map((event) => (
          <div
            key={event.id}
            className="rounded-lg border border-slate-700 p-3 text-sm"
          >
            <p>
              {event.config.relative_value?.category}:{" "}
              {event.config.relative_value?.target} · {event.snapshot.as_of}
            </p>
            <p className="text-emerald-300">
              {event.snapshot.signal?.direction_text}
            </p>
            <p className="text-slate-400">
              Residual score {event.snapshot.value.toFixed(3)} · Recorded{" "}
              {new Date(event.created_at).toLocaleString()}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}
