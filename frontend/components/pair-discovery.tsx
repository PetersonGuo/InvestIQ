"use client";
import { useState } from "react";
import { api } from "@/lib/stockassist";
import { PairConfig, PairSnapshot } from "@/lib/pairs";
type Candidate = {
	correlation: number;
	config: PairConfig;
	snapshot: PairSnapshot;
};
export default function PairDiscovery({
	select,
}: {
	select: (config: PairConfig, snapshot: PairSnapshot) => void;
}) {
	const [tickers, setTickers] = useState(
		"AAPL, MSFT, GOOGL, AMZN, META, NVDA",
	);
	const [minimum, setMinimum] = useState(0.8);
	const [lookback, setLookback] = useState(60);
	const [relationship, setRelationship] = useState("positive");
	const [cointegrated, setCointegrated] = useState(false);
	const [rows, setRows] = useState<Candidate[]>([]);
	const [message, setMessage] = useState("");
	const [busy, setBusy] = useState(false);
	const input = "rounded-lg border border-slate-700 bg-[#0c141e] p-2 text-sm";
	return (
		<section className="space-y-4 rounded-2xl border border-[#243141] bg-[#101923] p-6">
			<h2 className="font-semibold">Find correlated pairs</h2>
			<p className="text-sm text-slate-400">
				Scan 2–20 stocks using completed daily returns. The fitted hedge
				ratio uses prior closes. Correlation does not establish
				cointegration or guarantee mean reversion.
			</p>
			<form
				className="flex flex-wrap items-end gap-3"
				onSubmit={async (e) => {
					e.preventDefault();
					setBusy(true);
					setMessage("");
					setRows([]);
					try {
						const result = await api<{
							pairs: Candidate[];
							errors: { ticker: string; error: string }[];
						}>("pairs/discover", {
							method: "POST",
							body: JSON.stringify({
								tickers: tickers
									.split(/[\s,]+/)
									.filter(Boolean),
								min_correlation: minimum,
								lookback,
								threshold: 2,
								relationship,
								require_cointegration: cointegrated,
							}),
						});
						setRows(result.pairs);
						setMessage(
							`${result.pairs.length} matching pairs. ${result.errors.map((e) => `${e.ticker}: ${e.error}`).join(" ")}`,
						);
					} catch (e) {
						setMessage(
							e instanceof Error ? e.message : "Scan failed.",
						);
					} finally {
						setBusy(false);
					}
				}}
			>
				<label className="flex min-w-0 flex-1 flex-col gap-2 text-sm">
					Stock symbols
					<input
						className={input}
						value={tickers}
						onChange={(e) => setTickers(e.target.value)}
						required
					/>
				</label>
				<label className="flex flex-col gap-2 text-sm">
					Minimum correlation
					<input
						className={`${input} w-36`}
						type="number"
						min="0"
						max="1"
						step=".01"
						value={minimum}
						onChange={(e) => setMinimum(Number(e.target.value))}
						required
					/>
				</label>
				<label className="flex flex-col gap-2 text-sm">
					Trading days
					<input
						className={`${input} w-28`}
						type="number"
						min="20"
						max="250"
						value={lookback}
						onChange={(e) => setLookback(Number(e.target.value))}
						required
					/>
				</label>
				<button
					disabled={busy}
					className="rounded-lg bg-emerald-400 px-4 py-2 text-sm font-semibold text-emerald-950 disabled:opacity-40"
				>
					{busy ? "Scanning…" : "Find pairs"}
				</button>
			</form>
			<div className="flex flex-wrap items-center gap-4 text-sm">
				<label>
					Relationship{" "}
					<select
						className={input}
						value={relationship}
						onChange={(e) => setRelationship(e.target.value)}
					>
						<option value="positive">Positive correlation</option>
						<option value="negative">
							Inverse / negative correlation
						</option>
						<option value="either">Either direction</option>
					</select>
				</label>
				<label>
					<input
						type="checkbox"
						checked={cointegrated}
						onChange={(e) => setCointegrated(e.target.checked)}
					/>{" "}
					Require cointegration evidence at 5%
				</label>
			</div>
			<p className="text-xs text-slate-400">
				Engle–Granger uses prior log prices with an intercept and one
				ADF lag. Assumes I(1) series; scanning many pairs increases
				false discoveries. Inverse pairs can require both legs long or
				both short and are not market-neutral.
			</p>
			<p role="status" className="text-sm text-slate-400">
				{message}
			</p>
			<ul className="divide-y divide-slate-800">
				{rows.map((row) => (
					<li
						key={`${row.config.ticker_a}/${row.config.ticker_b}`}
						className="flex flex-wrap items-center justify-between gap-3 py-3 text-sm"
					>
						<div>
							<b>
								{row.config.ticker_a} / {row.config.ticker_b}
							</b>
							<p className="text-slate-400">
								Correlation {row.correlation.toFixed(3)} · z{" "}
								{row.snapshot.value.toFixed(2)} · hedge{" "}
								{row.config.hedge_ratio.toFixed(3)} ·{" "}
								{row.snapshot.source} · {row.snapshot.as_of}
							</p>
							<p className="text-emerald-300">
								{row.snapshot.signal
									? row.snapshot.signal.direction_text ||
										`Long ${row.snapshot.signal.long} / Short ${row.snapshot.signal.short}`
									: "Waiting for divergence beyond ±2 z"}
							</p>
						</div>
						<span className="text-xs text-slate-400">
							Cointegration:{" "}
							{row.snapshot.cointegration
								?.reject_no_cointegration_5_percent === true
								? "evidence at 5%"
								: row.snapshot.cointegration
											?.reject_no_cointegration_5_percent ===
									  false
									? "not established"
									: "unavailable"}
						</span>
						<button
							className="rounded-lg border border-emerald-500 px-3 py-2"
							onClick={() => select(row.config, row.snapshot)}
						>
							Configure alert
						</button>
					</li>
				))}
			</ul>
		</section>
	);
}
