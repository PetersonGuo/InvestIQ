import type { ValuationResult } from "@/components/peer-valuation";
export type StockList = {
  id: string;
  name: string;
  tickers: string[];
  version: number;
};
export type ListPlan = {
  id: string;
  active: boolean;
  running: boolean;
  next_run_at: string;
  last_finished: string | null;
  last_error: string | null;
  config: {
    source: string;
    stock_list_id: string;
    algorithms: string[];
    interval_hours: number;
    max_alerts: number;
    lookback: number;
    threshold: number;
    valuation_min_gap: number;
    valuation_peers: string[];
    min_correlation: number;
    sector: string;
    market: string;
    factors: string[];
    basket_size: number;
    model_limit: number;
  };
  last_result: {
    valuation?: ValuationResult | null;
    candidates: number;
    created: string[];
    managed_alerts: number;
    eligible?: number;
    evaluated?: number;
    errors: { ticker: string; error: string }[];
    skipped?: { ticker: string; reason: string }[];
  } | null;
};
export type TradeLeg = {
  ticker: string;
  side: string;
  weight: number;
  role: string;
};
export type Opportunity = {
  id: string;
  ticker_a: string;
  ticker_b: string;
  active: number;
  last_checked_at: string | null;
  last_error: string | null;
  relative_value: {
    category: string;
    target: string;
    references: { ticker: string; weight: number }[];
    market: string;
    sector: string;
    factors: string[];
    lookback: number;
    entry_z: number;
    [key: string]: unknown;
  };
  action: {
    state: string;
    label: string;
    legs: TradeLeg[];
    entry_z: number;
    exit_z: number;
    stop_z: number;
  };
  discovery: {
    list_name: string;
    list_version: number;
    reason: string;
    profiles: Record<
      string,
      {
        name?: string;
        industry?: string;
        category?: string;
        source?: string;
        fetched_at?: string;
      }
    >;
  } | null;
  latest: {
    as_of: string;
    source: string;
    value: number;
    baseline_start: string;
    baseline_end: string;
    relative_value: {
      fit_r2: number;
      score_type: string;
      exposures: {
        ticker: string;
        role: string;
        before_beta: number;
        after_beta: number;
      }[];
      hedge_legs: TradeLeg[];
      neutrality: string;
      event_gap_percent: number | null;
    };
  } | null;
};
export const familyName: Record<string, string> = {
  competitor: "Residual pair",
  basket: "Stock versus basket",
  lead_lag: "Daily lead-lag",
  same_company: "Same-company shares",
  etf: "ETF tracking",
  cross_asset: "Cross-asset proxy",
  event: "Event relationship",
};
