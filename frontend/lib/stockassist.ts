export type Bar = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};
export type Stock = {
  ticker: string;
  name: string;
  source: string;
  as_of: string;
  price: number;
  change_percent: number;
  bars: Bar[];
};
export type Alert = {
  id: string;
  ticker: string;
  direction: "above" | "below";
  threshold: number;
  active: number;
  triggered_at: string | null;
};
export type Portfolio = {
  cash_cents: number;
  positions: { ticker: string; quantity: number; cost_cents: number }[];
  orders: {
    id: string;
    ticker: string;
    side: string;
    quantity: number;
    price_cents: number;
    created_at: string;
    price_as_of: string;
  }[];
};
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (response.status === 204) return undefined as T;
  const body = await response.json();
  if (!response.ok)
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : "Invalid request. Check the fields and try again.",
    );
  return body;
}
export const money = (amount: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(
    amount,
  );
