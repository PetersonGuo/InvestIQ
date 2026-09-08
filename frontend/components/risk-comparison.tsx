export default function RiskComparison({
	rows,
	basis,
	benchmarkLabel,
}: {
	rows: Record<string, Record<string, number | null>>;
	basis?: string;
	benchmarkLabel?: string;
}) {
	const metrics = [
		["Return %", "total_return_percent"],
		["Max drawdown %", "max_drawdown_percent"],
		["Volatility %", "volatility_percent"],
		["Downside volatility %", "downside_volatility_percent"],
		["Sharpe", "sharpe_ratio"],
		["Sortino", "sortino_ratio"],
		["Historical VaR 95%", "var_95_percent"],
		["Historical CVaR 95%", "cvar_95_percent"],
		["Beta to SPY", "beta_to_spy"],
		["Tracking error %", "tracking_error_percent"],
		["Average gross exposure %", "average_gross_exposure_percent"],
		["Maximum gross exposure %", "max_gross_exposure_percent"],
		["Average net exposure %", "average_net_exposure_percent"],
		["Time invested %", "time_in_market_percent"],
	];
	return (
		<section className="mt-5">
			<h3 className="mb-3 font-semibold">
				Risk and benchmark comparison
			</h3>
			<div className="overflow-x-auto">
				<table className="w-full whitespace-nowrap text-left text-xs">
					<thead className="text-slate-400">
						<tr>
							<th className="p-2">Measure</th>
							<th className="p-2">Strategy</th>
							<th className="p-2">
								S&amp;P 500 · SPY buy &amp; hold
							</th>
							<th className="p-2">
								{benchmarkLabel || "Stock buy & hold"}
							</th>
						</tr>
					</thead>
					<tbody>
						{metrics.map(([label, key]) => (
							<tr key={key} className="border-t border-slate-800">
								<td className="p-2">{label}</td>
								{["strategy", "spy", "buy_and_hold"].map(
									(name) => (
										<td className="p-2" key={name}>
											{rows[name]?.[key] == null
												? "—"
												: rows[name][key]!.toFixed(2)}
										</td>
									),
								)}
							</tr>
						))}
					</tbody>
				</table>
			</div>
			<p className="mt-3 text-xs text-slate-400">
				{basis} Historical risk estimates can understate future losses.
				Gross exposure sums absolute position values; net exposure
				preserves their signs. SPY is a price-return proxy; dividends
				are not included.
			</p>
		</section>
	);
}
