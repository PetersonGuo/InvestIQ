"use client";
import { useEffect, useRef } from "react";
import { ColorType, createChart, LineSeries } from "lightweight-charts";
import { chartTime } from "@/lib/intervals";
export type EquityPoint = {
	time: string;
	equity: number;
	benchmark: number;
	spy?: number;
	drawdown_percent: number;
};
export default function EquityChart({
	points,
	benchmarkLabel,
}: {
	points: EquityPoint[];
	benchmarkLabel?: string;
}) {
	const ref = useRef<HTMLDivElement>(null);
	useEffect(() => {
		if (!ref.current) return;
		const chart = createChart(ref.current, {
			autoSize: true,
			height: 320,
			timeScale: {
				timeVisible: points[0]?.time.includes("T"),
				secondsVisible: true,
			},
			layout: {
				background: { type: ColorType.Solid, color: "#101923" },
				textColor: "#91a4b8",
			},
			grid: {
				vertLines: { color: "#1a2633" },
				horzLines: { color: "#1a2633" },
			},
		});
		const strategy = chart.addSeries(LineSeries, {
			color: "#42d6aa",
			lineWidth: 2,
			title: "Strategy",
		});
		const benchmark = chart.addSeries(LineSeries, {
			color: "#8b9bd3",
			lineWidth: 1,
			title: benchmarkLabel || "Buy & hold",
		});
		strategy.setData(
			points.map((p) => ({ time: chartTime(p.time), value: p.equity })),
		);
		benchmark.setData(
			points.map((p) => ({
				time: chartTime(p.time),
				value: p.benchmark,
			})),
		);
		if (points.every((p) => p.spy !== undefined)) {
			const spy = chart.addSeries(LineSeries, {
				color: "#f6c761",
				lineWidth: 2,
				title: "S&P 500 (SPY)",
			});
			spy.setData(
				points.map((p) => ({ time: chartTime(p.time), value: p.spy! })),
			);
		}
		chart.timeScale().fitContent();
		return () => chart.remove();
	}, [points, benchmarkLabel]);
	return (
		<div
			ref={ref}
			className="h-80"
			role="img"
			aria-label="Strategy equity compared with buy and hold"
		/>
	);
}
