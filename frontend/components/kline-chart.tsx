"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, ColorType, CandlestickSeries } from "lightweight-charts";
import { chartTime } from "@/lib/intervals";
import { api, type Bar } from "@/lib/stockassist";

export default function KLineChart({ symbol, bars, interval = "1d" }: { symbol: string; bars: Bar[]; interval?: string }) {
  const container = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState("");
  const retry = useRef<() => void>(() => {});
  useEffect(() => {
    if (!container.current) return;
    let data = [...bars], loading = false, exhausted = false, alive = true, failed = false;
    const controller = new AbortController();
    setStatus("");
    const chart = createChart(container.current, {
      autoSize: true, height: 360,
      layout: { background: { type: ColorType.Solid, color: "#101923" }, textColor: "#91a4b8", attributionLogo: true },
      grid: { vertLines: { color: "#1a2633" }, horzLines: { color: "#1a2633" } },
      rightPriceScale: { borderColor: "#283747" }, timeScale: { borderColor: "#283747", timeVisible: interval !== "1d", secondsVisible: interval.endsWith("s") },
    });
    const series = chart.addSeries(CandlestickSeries, {
      priceFormat: { type: "price", precision: interval === "1d" ? 2 : 4, minMove: interval === "1d" ? .01 : .0001 },
      upColor: "#42d6aa", downColor: "#f08080", wickUpColor: "#42d6aa", wickDownColor: "#f08080", borderVisible: false,
    });
    series.setData(data.map(bar => ({ ...bar, time: chartTime(bar.time) })));
    chart.timeScale().fitContent();
    async function loadOlder(force = false) {
      const range = chart.timeScale().getVisibleLogicalRange();
      if (!alive || loading || exhausted || failed || !data.length || !range || (!force && range.from > 15)) return;
      loading = true;
      setStatus(`Loading older ${interval === "1d" ? "daily" : interval} bars…`);
      try {
        const page = await api<{ bars: Bar[]; has_more: boolean }>(`stocks/${encodeURIComponent(symbol)}?days=250&interval=${interval}&before=${encodeURIComponent(data[0].time)}`, { signal: controller.signal });
        if (!alive) return;
        const older = page.bars.filter(bar => bar.time < data[0].time);
        const view = chart.timeScale().getVisibleLogicalRange();
        data = [...older, ...data];
        exhausted = !page.has_more || !older.length;
        series.setData(data.map(bar => ({ ...bar, time: chartTime(bar.time) })));
        if (view) chart.timeScale().setVisibleLogicalRange({ from: view.from + older.length, to: view.to + older.length });
        setStatus(exhausted ? "Beginning of available history" : `${data.length} ${interval === "1d" ? "daily" : interval} bars loaded`);
      } catch (error) {
        if (alive) { failed = true; setStatus(error instanceof Error ? error.message : "Could not load older history."); }
      } finally { loading = false; }
    }
    retry.current = () => { failed = false; void loadOlder(true); };
    const onRange = () => { void loadOlder(); };
    chart.timeScale().subscribeVisibleLogicalRangeChange(onRange);
    return () => { alive = false; controller.abort(); chart.timeScale().unsubscribeVisibleLogicalRangeChange(onRange); chart.remove(); };
  }, [symbol, bars, interval]);
  return <div>
    <div role="img" aria-label={`${symbol} ${interval === "1d" ? "daily" : interval} candlestick price chart. Latest close ${bars.at(-1)?.close ?? "unavailable"} dollars.`} ref={container} className="h-[360px] w-full" />
    <div className="flex items-center justify-between px-2 text-xs text-slate-400">
      <span aria-live="polite">{status || "Pan left or zoom out to load older bars"}</span>
      <button type="button" onClick={() => retry.current()} className="p-2 text-emerald-300">Load older history</button>
    </div>
  </div>;
}
