"use client";
import { useEffect, useState } from "react";
import { api, LiveQuote, money, Portfolio } from "@/lib/stockassist";
export default function LivePositions({
	positions,
	select,
}: {
	positions: Portfolio["positions"];
	select: (ticker: string) => void;
}) {
	const [quotes, setQuotes] = useState<Record<string, LiveQuote>>({});
	const symbols = positions.map((p) => p.ticker).join(",");
	useEffect(() => {
		let alive = true;
		let timer: ReturnType<typeof setTimeout>;
		const controller = new AbortController();
		async function refresh() {
			const results = await Promise.all(
				symbols
					.split(",")
					.filter(Boolean)
					.map(async (ticker) => {
						try {
							return await api<LiveQuote>(
								`stocks/${encodeURIComponent(ticker)}/quote`,
								{ signal: controller.signal },
							);
						} catch {
							return null;
						}
					}),
			);
			if (alive) {
				setQuotes(
					Object.fromEntries(
						results
							.filter((q): q is LiveQuote => !!q)
							.map((q) => [q.ticker, q]),
					),
				);
				timer = setTimeout(refresh, 3000);
			}
		}
		void refresh();
		return () => {
			alive = false;
			controller.abort();
			clearTimeout(timer);
		};
	}, [symbols]);
	return (
		<div className="overflow-x-auto">
			<table className="w-full text-left text-sm">
				<thead className="text-xs text-slate-500">
					<tr>
						<th>Symbol</th>
						<th>Shares</th>
						<th>Cost basis</th>
						<th>Live value (bid)</th>
						<th>Unrealized P&amp;L</th>
					</tr>
				</thead>
				<tbody>
					{positions.map((p) => {
						const q = quotes[p.ticker];
						const fresh =
							q &&
							q.connected &&
							!q.error &&
							q.market_data_type === 1 &&
							q.last_time &&
							Date.now() - Date.parse(q.last_time) < 60000;
						const value =
							fresh && q.bid ? q.bid * p.quantity : null;
						return (
							<tr
								key={p.ticker}
								className="border-t border-[#243141]"
							>
								<td className="py-3">
									<button
										className="hover:text-emerald-400"
										onClick={() => select(p.ticker)}
									>
										{p.ticker}
									</button>
								</td>
								<td>{p.quantity}</td>
								<td>{money(p.cost_cents / 100)}</td>
								<td>
									{value === null
										? "Unavailable"
										: money(value)}
								</td>
								<td>
									{value === null
										? "—"
										: money(value - p.cost_cents / 100)}
								</td>
							</tr>
						);
					})}
				</tbody>
			</table>
			<p className="mt-2 text-xs text-slate-500">
				Marks update from live bids. P&amp;L includes entry fees and
				excludes estimated exit fees.
			</p>
		</div>
	);
}
