"use client";
import { useEffect, useRef } from "react";
import {
  createChart,
  ColorType,
  LineSeries,
  LineStyle,
} from "lightweight-charts";
import type { PairConfig, PairSnapshot } from "@/lib/pairs";

export default function PairChart({
  snapshot,
  config,
}: {
  snapshot: PairSnapshot;
  config: PairConfig;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const chart = createChart(ref.current, {
      autoSize: true,
      height: 300,
      layout: {
        background: { type: ColorType.Solid, color: "#101923" },
        textColor: "#91a4b8",
      },
      grid: {
        vertLines: { color: "#1a2633" },
        horzLines: { color: "#1a2633" },
      },
    });
    const line = chart.addSeries(LineSeries, {
      color: "#42d6aa",
      lineWidth: 2,
      priceFormat: { type: "price", precision: 4, minMove: 0.0001 },
    });
    line.setData(snapshot.points);
    line.createPriceLine({
      price: config.threshold,
      color: "#e6b877",
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      axisLabelVisible: true,
      title: "Alert level",
    });
    if (config.condition === "outside" || config.condition === "inside")
      line.createPriceLine({
        price: -config.threshold,
        color: "#e6b877",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "Alert level",
      });
    if (config.metric === "zscore")
      line.createPriceLine({
        price: 0,
        color: "#586a82",
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: false,
        title: "Mean",
      });
    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [snapshot, config]);
  return (
    <div
      ref={ref}
      className="h-[300px]"
      role="img"
      aria-label={`${snapshot.ticker_a} and ${snapshot.ticker_b} ${snapshot.metric === "zscore" ? "spread z-score" : "price ratio"} history`}
    />
  );
}
