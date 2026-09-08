export type PairConfig = {
  ticker_a: string;
  ticker_b: string;
  metric: "zscore" | "ratio";
  condition: "above" | "below" | "outside" | "inside";
  threshold: number;
  lookback: number;
  hedge_ratio: number;
  repeat: boolean;
};
export type PairSnapshot = {
  ticker_a: string;
  ticker_b: string;
  metric: "zscore" | "ratio";
  value: number;
  ratio: number;
  z_score: number | null;
  spread: number;
  mean: number | null;
  std: number | null;
  source: string;
  as_of: string;
  matched: boolean;
  price_a: number;
  price_b: number;
  baseline_start: string | null;
  baseline_end: string | null;
  aligned_bars: number;
  points: { time: string; value: number }[];
};
export type PairAlert = PairConfig & {
  id: string;
  active: number;
  triggered_at: string | null;
  latest: PairSnapshot | null;
  last_error: string | null;
  last_checked_at: string | null;
};
export type PairEvent = {
  id: string;
  alert_id: string;
  created_at: string;
  as_of: string;
  config: PairConfig;
  snapshot: Omit<PairSnapshot, "points">;
};
export function conditionLabel(config: PairConfig) {
  const metric = config.metric === "zscore" ? "z" : "A ÷ B";
  if (config.condition === "outside")
    return `|${metric}| ≥ ${config.threshold}`;
  if (config.condition === "inside") return `|${metric}| ≤ ${config.threshold}`;
  return `${metric} ${config.condition === "above" ? "≥" : "≤"} ${config.threshold}`;
}
