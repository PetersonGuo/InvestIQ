"use client";
import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/stockassist";

type Config = { source: string; tickers?: string[]; interval_hours: number; max_alerts: number; lookback: number; min_correlation: number; threshold: number; relationship: string; require_cointegration: boolean; scan_code?: string; limit?: number; min_price?: number; max_price?: number; min_volume?: number };
type Schedule = { id: string; config: Config; active: boolean; running: boolean; next_run_at: string; last_finished: string | null; last_error: string | null; last_result: { candidates: number; created: string[]; managed_alerts: number; tickers: string[]; errors: { ticker: string; error: string }[] } | null };
const defaults: Config = { source: "scanner", interval_hours: 24, max_alerts: 5, lookback: 60, min_correlation: .8, threshold: 2, relationship: "positive", require_cointegration: true, scan_code: "HOT_BY_VOLUME", limit: 20, min_price: 5, max_price: 1000, min_volume: 100000 };
const field = "min-w-0 rounded-lg border border-slate-700 bg-[#0c141e] p-2 text-sm";
const button = "rounded-lg border border-slate-600 px-3 py-2 text-sm disabled:opacity-40";
const date = (value: string) => new Date(value).toLocaleString();
export default function PairSchedules() {
  const [rows, setRows] = useState<Schedule[]>([]);
  const [config, setConfig] = useState<Config>(defaults);
  const [symbols, setSymbols] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const refresh = useCallback(async () => setRows((await api<Schedule[]>("pairs/schedules")).filter(row => ["scanner", "symbols"].includes(row.config.source))), []);
  useEffect(() => { const load = () => { void refresh().catch(e => setMessage(e instanceof Error ? e.message : "Could not load schedules.")); }; load(); const timer = setInterval(load, 5000); return () => clearInterval(timer); }, [refresh]);
  const change = (key: keyof Config, value: string | number | boolean) => setConfig(c => ({ ...c, [key]: value }));
  async function mutate(path: string, method: string, body?: unknown) {
    setBusy(true); setMessage("");
    try { await api(path, { method, ...(body === undefined ? {} : { body: JSON.stringify(body) }) }); await refresh(); return true; }
    catch(e) { setMessage(e instanceof Error ? e.message : "Schedule request failed."); return false; }
    finally { setBusy(false); }
  }
  return <section aria-label="Scheduled pair discovery" className="space-y-4 rounded-2xl border border-[#243141] bg-[#101923] p-6">
    <h2 className="font-semibold">Automatic pair discovery</h2>
    <p className="text-sm text-slate-400">Find pairs on a saved schedule and automatically create repeating divergence alerts with fitted hedge ratios and trade directions. The C++ backend must stay running. Searches use completed daily bars; alerts appear in this app.</p>
    <form className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3" onSubmit={async e => { e.preventDefault(); if(await mutate(`pairs/schedules${editing ? `/${editing}` : ""}`, editing ? "PUT" : "POST", { ...config, ...(config.source === "symbols" ? { tickers: symbols.split(/[\s,]+/).filter(Boolean) } : {}) })) { setEditing(null); setMessage("Schedule saved. An active schedule runs shortly, then at its saved interval."); } }}>
      <label className="grid gap-2 text-sm">Scheduled stock source<select className={field} value={config.source} onChange={e => change("source", e.target.value)}><option value="scanner">Refresh from IBKR scanner each run</option><option value="symbols">My stock list</option></select></label>
      {config.source === "symbols" ? <label className="grid gap-2 text-sm">Scheduled stock symbols<input className={field} value={symbols} onChange={e => setSymbols(e.target.value)} required placeholder="Enter 2–20 symbols" /></label> : <>
        <label className="grid gap-2 text-sm">Scheduled scanner<select className={field} value={config.scan_code} onChange={e => change("scan_code", e.target.value)}><option value="HOT_BY_VOLUME">Most active by volume</option><option value="TOP_PERC_GAIN">Top percentage gainers</option><option value="TOP_PERC_LOSE">Top percentage losers</option></select></label>
        {([['limit','Stocks per scan',2,20,1],['min_price','Scanner minimum price',0,100000,.01],['max_price','Scanner maximum price',.01,100000,.01],['min_volume','Scanner minimum volume',0,1000000000,1]] as const).map(([key,label,min,max,step]) => <label key={key} className="grid gap-2 text-sm">{label}<input className={field} type="number" required min={min} max={max} step={step} value={config[key] ?? defaults[key]} onChange={e => change(key, Number(e.target.value))} /></label>)}
      </>}
      {([['interval_hours','Repeat every (hours)',1,168,1],['max_alerts','Maximum managed alerts',1,50,1],['lookback','Scheduled lookback (trading days)',20,250,1],['min_correlation','Scheduled minimum correlation',0,1,.01],['threshold','Scheduled entry z-score',.1,20,.1]] as const).map(([key,label,min,max,step]) => <label key={key} className="grid gap-2 text-sm">{label}<input className={field} type="number" required min={min} max={max} step={step} value={config[key]} onChange={e => change(key, Number(e.target.value))} /></label>)}
      <label className="grid gap-2 text-sm">Scheduled relationship<select className={field} value={config.relationship} onChange={e => change("relationship", e.target.value)}><option value="positive">Positive</option><option value="negative">Inverse / negative</option><option value="either">Either direction</option></select></label>
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={config.require_cointegration} onChange={e => change("require_cointegration", e.target.checked)} />Scheduled cointegration evidence at 5%</label>
      <div className="flex items-end gap-2"><button disabled={busy} className="rounded-lg bg-emerald-400 px-4 py-2 text-sm font-semibold text-emerald-950 disabled:opacity-40">{editing ? "Save schedule" : "Enable automatic discovery"}</button>{editing && <button type="button" className={button} onClick={() => { setEditing(null); setConfig(defaults); }}>Cancel edit</button>}</div>
    </form>
    <p className="text-xs text-slate-400">Each scan adds the strongest qualifying pairs up to the cap. Existing alerts keep their settings, even if later scans differ. Paused alerts count toward the cap; deleting one frees space and suppresses that pair for this schedule. Pausing or deleting a schedule leaves its alerts and history intact. Cointegration is statistical evidence, and inverse pairs may put both legs on the same side.</p>
    <p role="status" className="text-sm text-slate-300">{message}</p>
    {!rows.length && <p className="text-sm text-slate-400">No saved schedules yet.</p>}
    <ul className="divide-y divide-slate-800">{rows.map(row => <li key={row.id} className="space-y-2 py-4 text-sm">
      <div className="flex flex-wrap justify-between gap-3"><b>{row.config.source === "scanner" ? `IBKR ${row.config.scan_code}` : row.config.tickers?.join(", ")} · every {row.config.interval_hours}h</b><span>{!row.active ? "Paused" : row.running ? "Scanning…" : "Scheduled"}</span></div>
      <p className="text-slate-400">{row.active ? `Next run: ${date(row.next_run_at)}` : "Automatic runs paused"} · Last finished: {row.last_finished ? date(row.last_finished) : "Not run yet"}</p>
      {row.last_result && <p className="text-slate-300">{row.last_result.candidates} qualifying pairs · {row.last_result.created.length} alerts added last run · {row.last_result.managed_alerts}/{row.config.max_alerts} managed at last run. {row.last_result.created.join(", ")}</p>}
      {row.last_result?.errors.map(error => <p key={error.ticker} className="break-words text-amber-300">{error.ticker}: {error.error}</p>)}
      {row.last_error && <p className="break-words text-amber-300">{row.last_error} {row.active && "Retry scheduled in five minutes after failure."}</p>}
      <div className="flex flex-wrap gap-2"><button className={button} disabled={busy || row.running || !row.active} onClick={() => mutate(`pairs/schedules/${row.id}/run`, "POST")}>Run now</button><button className={button} disabled={busy} onClick={() => mutate(`pairs/schedules/${row.id}/state`, "POST", { active: !row.active })}>{row.active ? "Pause schedule" : "Resume schedule"}</button><button className={button} disabled={busy} onClick={() => { setEditing(row.id); setConfig({ ...defaults, ...row.config }); setSymbols(row.config.tickers?.join(", ") ?? ""); }}>Edit schedule</button><button className={button} disabled={busy} onClick={async () => { if(await mutate(`pairs/schedules/${row.id}`, "DELETE") && editing === row.id) setEditing(null); }}>Delete schedule</button></div>
    </li>)}</ul>
  </section>;
}
