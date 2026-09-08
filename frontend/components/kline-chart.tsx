"use client";

import { useEffect, useRef, useState } from "react";
import { createChart, ColorType, CandlestickSeries } from "lightweight-charts";
import { chartTime, intervalSeconds } from "@/lib/intervals";
import { api, type Bar, type LiveQuote } from "@/lib/stockassist";

export default function KLineChart({
	symbol,
	bars,
	interval = "1d",
	onQuote,
}: {
	symbol: string;
	bars: Bar[];
	interval?: string;
	onQuote?: (quote: LiveQuote | null) => void;
}) {
	const container = useRef<HTMLDivElement>(null);
	const [status, setStatus] = useState("");
	const [liveStatus, setLiveStatus] = useState("Connecting to live quotes…");
	const quoteCallback = useRef(onQuote);
	quoteCallback.current = onQuote;
	const follow = useRef<() => void>(() => {});
	const retry = useRef<() => void>(() => {});
	useEffect(() => {
		if (!container.current) return;
		let data = [...bars],
			loading = false,
			exhausted = false,
			alive = true,
			failed = false;
		const controller = new AbortController();
		setStatus("");
		const chart = createChart(container.current, {
			autoSize: true,
			height: 360,
			layout: {
				background: { type: ColorType.Solid, color: "#101923" },
				textColor: "#91a4b8",
				attributionLogo: true,
			},
			grid: {
				vertLines: { color: "#1a2633" },
				horzLines: { color: "#1a2633" },
			},
			rightPriceScale: { borderColor: "#283747" },
			timeScale: {
				borderColor: "#283747",
				timeVisible: interval !== "1d",
				secondsVisible: interval.endsWith("s"),
			},
		});
		const series = chart.addSeries(CandlestickSeries, {
			priceFormat: {
				type: "price",
				precision: interval === "1d" ? 2 : 4,
				minMove: interval === "1d" ? 0.01 : 0.0001,
			},
			upColor: "#42d6aa",
			downColor: "#f08080",
			wickUpColor: "#42d6aa",
			wickDownColor: "#f08080",
			borderVisible: false,
		});
		series.setData(
			data.map((bar) => ({ ...bar, time: chartTime(bar.time) })),
		);
		chart.timeScale().fitContent();
		follow.current = () => chart.timeScale().scrollToRealTime();
		let liveTimer: ReturnType<typeof setTimeout>;
		let lastObservation = "";
		async function refreshLive() {
			try {
				const quote = await api<LiveQuote>(
					`stocks/${encodeURIComponent(symbol)}/quote`,
					{ signal: controller.signal },
				);
				if (!alive) return;
				quoteCallback.current?.(quote);
				const age = quote.last_time
					? (Date.now() - Date.parse(quote.last_time)) / 1000
					: Infinity;
				if (
					quote.error ||
					quote.market_data_type !== 1 ||
					!quote.connected ||
					age > 60 ||
					!quote.last
				) {
					setLiveStatus(
						quote.error ||
							"Waiting for fresh live trades · history remains available",
					);
				} else {
					setLiveStatus(
						`${quote.source === "demo" ? "Demo quotes" : "Live IBKR"} · ${quote.last.toFixed(2)} · ${new Date(quote.last_time!).toLocaleTimeString()} · sampled candles`,
					);
					const observation = `${quote.last_time}:${quote.last}`;
					if (observation !== lastObservation) {
						lastObservation = observation;
						const seconds = intervalSeconds[interval];
						const timestamp = Date.parse(quote.last_time!);
						let time =
							interval === "1d"
								? new Intl.DateTimeFormat("en-CA", {
										timeZone: "America/New_York",
										year: "numeric",
										month: "2-digit",
										day: "2-digit",
									}).format(new Date(timestamp))
								: new Date(
										Math.floor(
											timestamp / (seconds * 1000),
										) *
											seconds *
											1000,
									).toISOString();
						const last = data.at(-1);
						if (last && interval !== "1d" && seconds >= 3600) {
							const anchor = Date.parse(last.time);
							if (
								timestamp >= anchor &&
								timestamp - anchor < 86400000
							)
								time = new Date(
									anchor +
										Math.floor(
											(timestamp - anchor) /
												(seconds * 1000),
										) *
											seconds *
											1000,
								).toISOString();
						}
						if (
							!last ||
							Date.parse(time) >= Date.parse(last.time)
						) {
							const same =
								last &&
								Date.parse(time) === Date.parse(last.time);
							const bar = same
								? {
										...last,
										high: Math.max(last.high, quote.last),
										low: Math.min(last.low, quote.last),
										close: quote.last,
									}
								: {
										time,
										open: quote.last,
										high: quote.last,
										low: quote.last,
										close: quote.last,
										volume: 0,
									};
							if (same) data[data.length - 1] = bar;
							else data.push(bar);
							series.update({
								...bar,
								time: chartTime(bar.time),
							});
						}
					}
				}
			} catch (error) {
				if (alive) {
					quoteCallback.current?.(null);
					setLiveStatus(
						error instanceof Error
							? error.message
							: "Live quotes disconnected",
					);
				}
			} finally {
				if (alive) liveTimer = setTimeout(refreshLive, 1000);
			}
		}
		void refreshLive();
		async function loadOlder(force = false) {
			const range = chart.timeScale().getVisibleLogicalRange();
			if (
				!alive ||
				loading ||
				exhausted ||
				failed ||
				!data.length ||
				!range ||
				(!force && range.from > 15)
			)
				return;
			loading = true;
			setStatus(
				`Loading older ${interval === "1d" ? "daily" : interval} bars…`,
			);
			try {
				const page = await api<{ bars: Bar[]; has_more: boolean }>(
					`stocks/${encodeURIComponent(symbol)}?days=250&interval=${interval}&before=${encodeURIComponent(data[0].time)}`,
					{ signal: controller.signal },
				);
				if (!alive) return;
				const older = page.bars.filter(
					(bar) => bar.time < data[0].time,
				);
				const view = chart.timeScale().getVisibleLogicalRange();
				data = [...older, ...data];
				exhausted = !page.has_more || !older.length;
				series.setData(
					data.map((bar) => ({ ...bar, time: chartTime(bar.time) })),
				);
				if (view)
					chart
						.timeScale()
						.setVisibleLogicalRange({
							from: view.from + older.length,
							to: view.to + older.length,
						});
				setStatus(
					exhausted
						? "Beginning of available history"
						: `${data.length} ${interval === "1d" ? "daily" : interval} bars loaded`,
				);
			} catch (error) {
				if (alive) {
					failed = true;
					setStatus(
						error instanceof Error
							? error.message
							: "Could not load older history.",
					);
				}
			} finally {
				loading = false;
			}
		}
		retry.current = () => {
			failed = false;
			void loadOlder(true);
		};
		const onRange = () => {
			void loadOlder();
		};
		chart.timeScale().subscribeVisibleLogicalRangeChange(onRange);
		return () => {
			alive = false;
			clearTimeout(liveTimer);
			controller.abort();
			chart.timeScale().unsubscribeVisibleLogicalRangeChange(onRange);
			chart.remove();
		};
	}, [symbol, bars, interval]);
	return (
		<div>
			<div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2 text-xs text-slate-400">
				<span aria-live="polite">{liveStatus}</span>
				<button
					className="text-emerald-300"
					onClick={() => follow.current()}
				>
					Follow live
				</button>
			</div>
			<div
				role="img"
				aria-label={`${symbol} ${interval === "1d" ? "daily" : interval} candlestick price chart. Latest close ${bars.at(-1)?.close ?? "unavailable"} dollars.`}
				ref={container}
				className="h-[360px] w-full"
			/>
			<div className="flex items-center justify-between px-2 text-xs text-slate-400">
				<span aria-live="polite">
					{status || "Pan left or zoom out to load older bars"}
				</span>
				<button
					type="button"
					onClick={() => retry.current()}
					className="p-2 text-emerald-300"
				>
					Load older history
				</button>
			</div>
		</div>
	);
}
