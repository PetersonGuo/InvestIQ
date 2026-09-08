"use client";
import { useEffect, useState } from 'react';
import { api } from '@/lib/stockassist';
type Tick = { time: string; price: number; size: number; exchange: string; conditions: string; past_limit: boolean; unreported: boolean };
export default function TradeTape({ symbol }: { symbol: string }) {
  const [rows, setRows] = useState<Tick[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [loadedSymbol, setLoadedSymbol] = useState('');
  useEffect(() => { setRows([]); setError(''); }, [symbol]);
  async function load() {
    setBusy(true); setError('');
    try {
      const data = await api<{ ticks: Tick[] }>(`stocks/${encodeURIComponent(symbol)}/ticks`);
      setRows(data.ticks); setLoadedSymbol(symbol);
    } catch (error) { setError(error instanceof Error ? error.message : 'Could not load trades.'); }
    finally { setBusy(false); }
  }
  const visible = loadedSymbol === symbol ? rows : [];
  return <details className="rounded-2xl border border-[#243141] bg-[#101923] p-5">
    <summary className="cursor-pointer font-semibold">Individual trades · IBKR Time &amp; Sales</summary>
    <p className="my-3 text-xs text-slate-400">Historical trade snapshot, regular session. IBKR timestamps have one-second resolution; multiple trades in the same second remain separate. Cached for 60 seconds. These records are not candles or a live stream.</p>
    <div className="flex gap-4 text-sm">
      <button disabled={busy} onClick={() => void load()} className="rounded bg-emerald-400 px-3 py-2 text-slate-950 disabled:opacity-40">{busy ? 'Loading trades…' : 'Load latest trades'}</button>
      {!!visible.length && <button onClick={() => {
        const url = URL.createObjectURL(new Blob([JSON.stringify({ticker:symbol,source:'ibkr',ticks:visible},null,2)],{type:'application/json'}));
        const link = document.createElement('a'); link.href=url; link.download=`${symbol}-trades.json`; link.click(); URL.revokeObjectURL(url);
      }} className="text-emerald-300">Export trades</button>}
    </div>
    {error && <p className="mt-3 text-sm text-red-300">{error}</p>}
    {loadedSymbol === symbol && !visible.length && !busy && <p className="mt-3 text-sm text-slate-400">No trades returned for this session.</p>}
    {!!visible.length && <><p className="my-3 text-xs text-slate-400">{visible.length} trades · {visible[0].time} to {visible.at(-1)?.time} · UTC</p>
      <div className="max-h-72 overflow-auto"><table className="w-full whitespace-nowrap text-left text-xs"><thead><tr>{['Time · UTC','Price','Size','Exchange','Conditions'].map(h=><th key={h} className="p-2">{h}</th>)}</tr></thead><tbody>{[...visible].reverse().map((tick,i)=><tr key={i} className="border-t border-[#243141]"><td className="p-2">{tick.time}</td><td className="p-2">{tick.price.toLocaleString("en-US", {minimumFractionDigits:2, maximumFractionDigits:6})}</td><td className="p-2">{tick.size}</td><td className="p-2">{tick.exchange}</td><td className="p-2">{tick.conditions}{tick.unreported ? ' · unreported' : ''}{tick.past_limit ? ' · past limit' : ''}</td></tr>)}</tbody></table></div></>}
  </details>;
}
