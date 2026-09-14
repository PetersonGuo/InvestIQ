"use client";
import { useState } from "react";
import { api } from "@/lib/stockassist";
export type ValuationResult = {
  model: string;
  as_of: string;
  computed_at: string;
  notes: string[];
  errors: { ticker: string; error: string }[];
  rows: {
    ticker: string;
    name: string;
    status: string;
    reason: string;
    group: string;
    industry: string;
    multiple: number;
    expected_multiple?: number;
    lower_multiple?: number;
    upper_multiple?: number;
    baseline_multiple?: number;
    gap?: number;
    period_end: string;
    price_as_of: string;
    source: string;
    price_source: string;
    peers: string[];
    valuation_direction?: string;
    confirmation_error?: string;
    contributions?: {
      feature: string;
      value: number;
      peer_mean: number;
      log_multiple_contribution: number;
    }[];
    evidence: {
      tag: string;
      val: number;
    unit?: string;
      start?: string;
      end: string;
      filed: string;
      filing_url?: string;
    }[];
    action: {
      state: string;
      label: string;
      legs: { ticker: string; side: string; weight: number; role: string }[];
    };
  }[];
};
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
export default function PeerValuation({
  result,
  unavailable = false,
}: {
  result: ValuationResult | null | undefined;
  unavailable?: boolean;
}) {
  const [events, setEvents] = useState<
    {
      id: string;
      created_at: string;
      result: ValuationResult["rows"][number];
    }[]
  >([]);
  const [eventError, setEventError] = useState("");
  return (
    <section
      aria-label="Peer relative valuation"
      className="space-y-4 rounded-xl border border-indigo-400/30 bg-[#101a27] p-5"
    >
      <h2 className="text-lg font-semibold">
        Peer-relative fundamental valuation
      </h2>
      <p className="text-sm text-slate-400">
        Annual earnings multiples explained by growth, profitability, cash
        generation and liabilities/assets. At least 11 eligible companies in the
        same SEC industry group are needed. Your list and optional reference
        peers form the estimation universe.
      </p>
      {!result ? (
        <p className="text-sm">
          Enable Peer fundamental valuation in your list strategies, then run a
          scan. Missing or insufficient data will be explained here.
        </p>
      ) : (
        <>
          <p className="text-sm">
            As of {result.as_of} · {result.rows.length} companies with usable
            data · {result.rows.filter((r) => r.status === "estimated").length}{" "}
            estimates ·{" "}
            {unavailable
              ? 0
              : result.rows.filter((r) => r.action.state === "entry")
                  .length}{" "}
            confirmed candidates
          </p>
          {unavailable && (
            <p className="text-amber-300">
              Schedule paused, scan failed, or results could not be refreshed.
              Previous estimates remain for reference; no entry actions are
              shown.
            </p>
          )}
          {result.errors.map((e, i) => (
            <p key={i} className="text-sm text-amber-300">
              {e.ticker}: {e.error}
            </p>
          ))}
          {result.rows.map((row) => (
            <article
              aria-label={`${row.ticker} valuation`}
              key={row.ticker}
              className="space-y-3 rounded-lg border border-slate-700 p-4"
            >
              <div className="flex flex-wrap justify-between gap-2">
                <h3 className="font-semibold">
                  {row.ticker} · {row.name}
                </h3>
                <span className="text-sm">
                  {row.status.replaceAll("_", " ")}
                </span>
              </div>
              <p
                className={
                  row.action.state === "entry" && !unavailable
                    ? "text-emerald-300"
                    : "text-slate-300"
                }
              >
                {unavailable
                  ? "No action — previous estimate only"
                  : row.action.label}
              </p>
              {row.expected_multiple != null && (
                <p className="text-sm">
                  Actual P/E {row.multiple.toFixed(1)}× · Estimated{" "}
                  {row.expected_multiple.toFixed(1)}× · Empirical range{" "}
                  {row.lower_multiple?.toFixed(1)}–
                  {row.upper_multiple?.toFixed(1)}× ·{" "}
                  {Math.abs((row.gap ?? 0) * 100).toFixed(1)}%{" "}
                  {(row.gap ?? 0) < 0 ? "discount" : "premium"} to estimate
                </p>
              )}
              <p className="text-sm text-slate-400">{row.reason}</p>
              {row.confirmation_error && (
                <p className="text-sm text-amber-300">
                  Price confirmation unavailable: {row.confirmation_error}
                </p>
              )}
              {!unavailable && row.action.state === "entry" && (
                <div className="grid gap-3 sm:grid-cols-2">
                  {["long", "short"].map((side) => (
                    <div key={side}>
                      <h4 className="font-semibold">
                        {side === "long" ? "BUY / LONG" : "SELL SHORT"}
                      </h4>
                      {row.action.legs
                        .filter((l) => l.side === side)
                        .map((l) => (
                          <p key={l.ticker} className="text-sm">
                            {l.ticker} · {pct(Math.abs(l.weight))} gross ·{" "}
                            {l.role}
                          </p>
                        ))}
                    </div>
                  ))}
                </div>
              )}
              <details>
                <summary className="cursor-pointer text-sm text-indigo-300">
                  Why this valuation? Peers, drivers & filings
                </summary>
                <div className="mt-3 space-y-3 text-sm">
                  <p>
                    Industry group {row.group}: {row.industry}. Peers:{" "}
                    {row.peers.join(", ") || "None eligible"}. The target issuer
                    is excluded from fitting and uncertainty estimation.
                  </p>
                  <p>
                    Annual period ended {row.period_end}. Price dated{" "}
                    {row.price_as_of}. Sources: {row.source} /{" "}
                    {row.price_source}.
                  </p>
                  {row.baseline_multiple != null && (
                    <p>
                      Peer baseline {row.baseline_multiple.toFixed(1)}×.
                      Contributions below add to the log of that multiple; they
                      are fitted associations, not causal effects.
                    </p>
                  )}
                  <div className="overflow-x-auto">
                    <table className="w-full text-left">
                      <thead>
                        <tr>
                          <th>Driver</th>
                          <th>Company</th>
                          <th>Peer mean</th>
                          <th>Log contribution</th>
                        </tr>
                      </thead>
                      <tbody>
                        {row.contributions?.map((c) => (
                          <tr key={c.feature}>
                            <td className="py-1 pr-3">{c.feature}</td>
                            <td className="pr-3">{pct(c.value)}</td>
                            <td className="pr-3">{pct(c.peer_mean)}</td>
                            <td>{c.log_multiple_contribution.toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <ul className="space-y-2 break-words">
                    {row.evidence.map((f, i) => (
                      <li key={i}>
                        {f.tag}: {f.val.toLocaleString()} {f.unit} · period{" "}
                        {f.start ? `${f.start} to ` : ""}
                        {f.end} · filed {f.filed}{" "}
                        {f.filing_url?.startsWith("https://www.sec.gov/") && (
                          <a
                            className="text-indigo-300 underline"
                            href={f.filing_url}
                            target="_blank"
                            rel="noreferrer"
                          >
                            Open filing
                          </a>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              </details>
            </article>
          ))}
          <details>
            <summary className="cursor-pointer text-sm">
              Model assumptions and current limits
            </summary>
            <ul className="mt-3 space-y-2 text-sm text-slate-400">
              {result.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          </details>
        </>
      )}
      <details
        onToggle={(e) => {
          if (e.currentTarget.open)
            void api<typeof events>("pairs/valuation/events")
              .then(setEvents)
              .catch((e) => setEventError(e.message));
        }}
      >
        <summary className="cursor-pointer text-sm">
          Persistent valuation alert history (all lists)
        </summary>
        {eventError && <p className="text-amber-300">{eventError}</p>}
        {!events.length && (
          <p className="mt-2 text-sm text-slate-400">
            No recorded confirmed valuation entries.
          </p>
        )}
        {events.map((e) => (
          <p key={e.id} className="mt-2 text-sm">
            {e.created_at} · {e.result.ticker} · Historical entry candidate; not
            a current trade instruction.
          </p>
        ))}
      </details>
      <p className="text-xs text-slate-400">
        Research screen, not a fundamental-strategy backtest or an order. Review
        business comparability, corporate actions, borrow and costs. A lower
        valuation can be justified by risks this model does not capture.
      </p>
    </section>
  );
}
