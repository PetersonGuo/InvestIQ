"use client";
import Link from "next/link";
import { FormEvent, useCallback, useEffect, useState } from "react";
import {
	Bell,
	ChartCandlestick,
	GitCompareArrows,
	Pause,
	Play,
	Trash2,
} from "lucide-react";
import { api, money } from "@/lib/stockassist";
import {
	PairAlert,
	PairConfig,
	PairEvent,
	PairSnapshot,
	conditionLabel,
} from "@/lib/pairs";
import PairChart from "./pair-chart";
import PairDiscovery from "./pair-discovery";

const initial: PairConfig = {
	ticker_a: "AAPL",
	ticker_b: "MSFT",
	metric: "zscore",
	condition: "outside",
	threshold: 2,
	lookback: 60,
	hedge_ratio: 1,
	repeat: false,
};
const field =
	"w-full rounded-lg border border-[#2b3948] bg-[#0c141e] px-3 py-2.5 text-sm text-slate-100 outline-none focus:border-emerald-400";
const panel = "rounded-2xl border border-[#243141] bg-[#101923] p-6";
const button =
	"rounded-lg bg-emerald-400 px-4 py-2.5 text-sm font-semibold text-[#09251e] hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-40";
const number = (value: number) =>
	value.toLocaleString("en-US", { maximumFractionDigits: 4 });

export default function PairWorkspace() {
	const [config, setConfig] = useState<PairConfig>(initial);
	const [preview, setPreview] = useState<PairSnapshot | null>(null);
	const [alerts, setAlerts] = useState<PairAlert[]>([]);
	const [events, setEvents] = useState<PairEvent[]>([]);
	const [editing, setEditing] = useState<string | null>(null);
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState("");
	const [notice, setNotice] = useState("");
	const [refreshError, setRefreshError] = useState("");
	const refresh = useCallback(async () => {
		const [rows, history] = await Promise.all([
			api<PairAlert[]>("pairs/alerts"),
			api<PairEvent[]>("pairs/events"),
		]);
		setAlerts(rows);
		setEvents(history);
		setRefreshError("");
	}, []);
	useEffect(() => {
		let alive = true;
		const update = () => {
			refresh().catch((e) => {
				if (alive) setRefreshError(e.message);
			});
		};
		update();
		const timer = setInterval(update, 10000);
		return () => {
			alive = false;
			clearInterval(timer);
		};
	}, [refresh]);

	function change(update: Partial<PairConfig>) {
		setConfig((current) => ({ ...current, ...update }));
		setPreview(null);
		setNotice("");
	}
	async function inspect(event: FormEvent) {
		event.preventDefault();
		setBusy(true);
		setError("");
		setNotice("");
		try {
			setPreview(
				await api<PairSnapshot>("pairs/preview", {
					method: "POST",
					body: JSON.stringify(config),
				}),
			);
		} catch (e) {
			setError(e instanceof Error ? e.message : "Pair preview failed.");
		} finally {
			setBusy(false);
		}
	}
	async function mutate(action: () => Promise<unknown>, message: string) {
		setBusy(true);
		setError("");
		setNotice("");
		try {
			await action();
			await refresh();
			setNotice(message);
		} catch (e) {
			setError(
				e instanceof Error
					? e.message
					: "Could not update the pair alert.",
			);
		} finally {
			setBusy(false);
		}
	}
	function save() {
		void mutate(async () => {
			await api(editing ? `pairs/alerts/${editing}` : "pairs/alerts", {
				method: editing ? "PUT" : "POST",
				body: JSON.stringify(config),
			});
			setEditing(null);
		}, "Pair alert saved. The background worker will check it on its next cycle.");
	}
	function edit(alert: PairAlert) {
		const next: PairConfig = {
			ticker_a: alert.ticker_a,
			ticker_b: alert.ticker_b,
			metric: alert.metric,
			condition: alert.condition,
			threshold: alert.threshold,
			lookback: alert.lookback,
			hedge_ratio: alert.hedge_ratio,
			repeat: alert.repeat,
		};
		setConfig(next);
		setPreview(null);
		setEditing(alert.id);
		setNotice("");
		window.scrollTo({ top: 0, behavior: "smooth" });
	}
	return (
		<main className="min-h-screen bg-[#0a111a] text-slate-200">
			<header className="border-b border-[#233040] bg-[#0d1620]">
				<div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-4 px-5 py-5 lg:px-10">
					<Link
						href="/"
						className="flex items-center gap-3 text-xl font-semibold"
					>
						<ChartCandlestick className="text-emerald-400" />
						StockAssist
					</Link>
					<nav className="flex flex-wrap gap-5 text-sm">
						<Link
							href="/"
							className="text-slate-400 hover:text-white"
						>
							Market overview
						</Link>
						<Link
							href="/research"
							className="text-slate-400 hover:text-white"
						>
							Research & backtest
						</Link>
						<Link
							href="/pairs"
							aria-current="page"
							className="text-emerald-400"
						>
							Pair alerts
						</Link>
					</nav>
				</div>
			</header>
			<div className="mx-auto max-w-[1440px] space-y-6 px-5 py-8 lg:px-10">
				<div>
					<p className="mb-2 text-xs uppercase tracking-[.2em] text-emerald-400">
						Two stocks, one relationship
					</p>
					<h1 className="text-3xl font-semibold tracking-tight">
						Pair trading alerts
					</h1>
					<p className="mt-2 text-sm text-slate-400">
						Monitor divergence or convergence using aligned daily
						closes.
					</p>
				</div>
				<PairDiscovery
					select={(next, snapshot) => {
						setConfig(next);
						setPreview(snapshot);
						setEditing(null);
						setNotice(
							"Pair selected. Review the signal and save the alert below.",
						);
					}}
				/>
				{(error || refreshError) && (
					<div
						role="alert"
						aria-label="Pair alert error"
						className="rounded-xl border border-red-400/30 bg-red-400/10 p-4 text-sm text-red-200"
					>
						{error || refreshError}
					</div>
				)}
				{notice && (
					<div
						role="status"
						className="rounded-xl bg-emerald-400/10 p-4 text-sm text-emerald-200"
					>
						{notice}
					</div>
				)}
				<div className="grid items-start gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
					<section className={panel}>
						<h2 className="mb-5 flex items-center gap-2 font-semibold">
							<GitCompareArrows
								size={18}
								className="text-emerald-400"
							/>
							{editing ? "Edit pair alert" : "Set up a pair"}
						</h2>
						<form className="space-y-4" onSubmit={inspect}>
							<div className="grid grid-cols-2 gap-3">
								<label className="text-xs text-slate-400">
									Stock A
									<input
										className={`${field} mt-1 uppercase`}
										value={config.ticker_a}
										onChange={(e) =>
											change({
												ticker_a: e.target.value
													.toUpperCase()
													.trim(),
											})
										}
										maxLength={15}
										pattern="[A-Z][A-Z0-9.\-]{0,14}"
										required
										disabled={busy}
									/>
								</label>
								<label className="text-xs text-slate-400">
									Stock B
									<input
										className={`${field} mt-1 uppercase`}
										value={config.ticker_b}
										onChange={(e) =>
											change({
												ticker_b: e.target.value
													.toUpperCase()
													.trim(),
											})
										}
										maxLength={15}
										pattern="[A-Z][A-Z0-9.\-]{0,14}"
										required
										disabled={busy}
									/>
								</label>
							</div>
							<label className="block text-xs text-slate-400">
								Metric
								<select
									className={`${field} mt-1`}
									value={config.metric}
									disabled={busy}
									onChange={(e) =>
										change(
											e.target.value === "ratio"
												? {
														metric: "ratio",
														condition: "above",
														threshold: 1,
													}
												: {
														metric: "zscore",
														condition: "outside",
														threshold: 2,
													},
										)
									}
								>
									<option value="zscore">
										Log-spread z-score
									</option>
									<option value="ratio">
										Price ratio · A ÷ B
									</option>
								</select>
							</label>
							{config.metric === "zscore" && (
								<div className="grid grid-cols-2 gap-3">
									<label className="text-xs text-slate-400">
										Lookback · sessions
										<input
											className={`${field} mt-1`}
											type="number"
											min="20"
											max="250"
											step="1"
											value={config.lookback}
											disabled={busy}
											required
											onChange={(e) =>
												change({
													lookback: Number(
														e.target.value,
													),
												})
											}
										/>
									</label>
									<label className="text-xs text-slate-400">
										Log-spread weight β
										<input
											className={`${field} mt-1`}
											type="number"
											min="-100"
											max="100"
											step="any"
											value={config.hedge_ratio}
											disabled={busy}
											required
											onChange={(e) =>
												change({
													hedge_ratio: Number(
														e.target.value,
													),
												})
											}
										/>
									</label>
								</div>
							)}
							<label className="block text-xs text-slate-400">
								Notify when
								<select
									className={`${field} mt-1`}
									value={config.condition}
									disabled={busy}
									onChange={(e) =>
										change({
											condition: e.target
												.value as PairConfig["condition"],
										})
									}
								>
									<option value="above">
										At or above the level
									</option>
									<option value="below">
										At or below the level
									</option>
									{config.metric === "zscore" && (
										<>
											<option value="outside">
												Outside ± level · divergence
											</option>
											<option value="inside">
												Inside ± level · convergence
											</option>
										</>
									)}
								</select>
							</label>
							<label className="block text-xs text-slate-400">
								Alert level
								<input
									className={`${field} mt-1`}
									type="number"
									step="any"
									min={
										config.metric === "ratio"
											? 0.000001
											: ["outside", "inside"].includes(
														config.condition,
												  )
												? 0
												: -20
									}
									max={
										config.metric === "ratio" ? 100000 : 20
									}
									disabled={busy}
									required
									value={config.threshold}
									onChange={(e) =>
										change({
											threshold: Number(e.target.value),
										})
									}
								/>
							</label>
							<label className="flex items-start gap-2 text-xs leading-relaxed text-slate-400">
								<input
									className="mt-0.5 accent-emerald-400"
									type="checkbox"
									checked={config.repeat}
									disabled={busy}
									onChange={(e) =>
										change({ repeat: e.target.checked })
									}
								/>
								Repeat after the condition clears on a later
								daily bar
							</label>
							<button
								className={`${button} w-full`}
								disabled={busy}
							>
								{busy ? "Working…" : "Preview pair"}
							</button>
							<button
								type="button"
								className="w-full rounded-lg border border-[#3c5266] px-4 py-2.5 text-sm hover:bg-[#223044] disabled:cursor-not-allowed disabled:opacity-40"
								disabled={busy || !preview}
								onClick={save}
							>
								{editing
									? "Save & rearm pair alert"
									: "Create pair alert"}
							</button>
							{editing && (
								<button
									type="button"
									className="w-full text-xs text-slate-400"
									onClick={() => {
										setEditing(null);
										setPreview(null);
									}}
								>
									Cancel editing
								</button>
							)}
						</form>
						<p className="mt-4 text-xs leading-relaxed text-slate-500">
							{config.repeat
								? "Repeats on a new false-to-true condition. Staying beyond the level does not send duplicate alerts."
								: "One-shot: pauses after its first trigger. Rearm it to watch again."}{" "}
							A condition already met when armed triggers on the
							next worker check.
						</p>
					</section>
					<div className="min-w-0 space-y-6">
						<section className={panel}>
							<h2 className="mb-4 font-semibold">Pair preview</h2>
							{!preview ? (
								<p className="py-16 text-center text-sm text-slate-500">
									Choose two stocks and preview their
									relationship before creating an alert.
								</p>
							) : (
								<>
									<div className="mb-5 flex flex-wrap justify-between gap-3">
										<div>
											<h3 className="text-xl font-semibold">
												{preview.ticker_a} /{" "}
												{preview.ticker_b}
											</h3>
											<p className="mt-1 text-xs text-slate-400">
												{preview.source === "demo"
													? "SYNTHETIC DEMO"
													: preview.source.toUpperCase()}{" "}
												· Completed close{" "}
												{preview.as_of}
											</p>
										</div>
										<span
											className={`self-start rounded-full px-3 py-1.5 text-xs ${preview.matched ? "bg-amber-400/15 text-amber-200" : "bg-slate-400/10 text-slate-300"}`}
										>
											{preview.matched
												? "Condition currently met"
												: "Condition not met"}
										</span>
									</div>
									<div className="mb-5 grid grid-cols-3 gap-3">
										<div>
											<p className="text-xs text-slate-400">
												{preview.metric === "zscore"
													? "Spread z-score"
													: "Price ratio"}
											</p>
											<p className="mt-2 text-2xl font-semibold text-emerald-300">
												{number(preview.value)}
											</p>
										</div>
										<div>
											<p className="text-xs text-slate-400">
												{preview.ticker_a} close
											</p>
											<p className="mt-2 text-lg">
												{money(preview.price_a)}
											</p>
										</div>
										<div>
											<p className="text-xs text-slate-400">
												{preview.ticker_b} close
											</p>
											<p className="mt-2 text-lg">
												{money(preview.price_b)}
											</p>
										</div>
									</div>
									{preview.signal && (
										<p className="text-sm text-emerald-300">
											Entry signal:{" "}
											{preview.signal.direction_text ||
												`Long ${preview.signal.long} / Short ${preview.signal.short}`}
											. {preview.signal.reason}
										</p>
									)}
									{preview.cointegration && (
										<div className="text-xs text-slate-400">
											<p>
												Cointegration:{" "}
												{preview.cointegration
													.reject_no_cointegration_5_percent ===
												true
													? "evidence at 5%"
													: preview.cointegration
																.reject_no_cointegration_5_percent ===
														  false
														? "not established"
														: "test unavailable"}{" "}
												· ADF{" "}
												{preview.cointegration.test_statistic?.toFixed(
													3,
												) ?? "—"}{" "}
												· 5% critical{" "}
												{preview.cointegration.critical_5_percent?.toFixed(
													3,
												) ?? "—"}{" "}
												· fitted hedge{" "}
												{preview.cointegration.fitted_hedge_ratio?.toFixed(
													3,
												) ?? "—"}{" "}
												· half-life{" "}
												{preview.cointegration.half_life_bars?.toFixed(
													1,
												) ?? "—"}{" "}
												trading days
											</p>
											<p>
												{
													preview.cointegration
														.assumptions
												}
											</p>
											<p>
												Spread:{" "}
												{preview.deviation?.replaceAll(
													"_",
													" ",
												)}
											</p>
										</div>
									)}
									<button
										className="text-xs text-emerald-300"
										onClick={() =>
											change({
												ticker_a: config.ticker_b,
												ticker_b: config.ticker_a,
												hedge_ratio:
													1 / config.hedge_ratio,
											})
										}
									>
										Reverse pair order
									</button>
									<PairChart
										snapshot={preview}
										config={config}
									/>
									<p className="mt-4 text-xs text-slate-400">
										Alert rule: {conditionLabel(config)}
										{preview.baseline_start &&
											` · Baseline ${preview.baseline_start} to ${preview.baseline_end}`}
									</p>
								</>
							)}
						</section>
						<section className={panel}>
							<h2 className="mb-3 text-sm font-semibold">
								How this alert is calculated
							</h2>
							<p className="text-xs leading-relaxed text-slate-400">
								{config.metric === "zscore"
									? `Spread = ln(${config.ticker_a || "A"}) − ${config.hedge_ratio} × ln(${config.ticker_b || "B"}). The latest spread is compared with the mean and population standard deviation of the preceding ${config.lookback} shared sessions, excluding the latest close. β is the configured spread weight. Discovery supplies a fitted weight; the cointegration test independently refits its own hedge on the prior window.`
									: "The price ratio is Stock A’s close divided by Stock B’s close on the same date. A ratio threshold is independent of any spread weight."}
							</p>
							<p className="mt-3 text-xs leading-relaxed text-slate-500">
								The background worker checks about every 30
								seconds while the API runs. Alerts use completed
								daily closes, not intraday quotes. Mismatched
								dates, stale history, or an undefined z-score
								pause evaluation and show an error.
								Notifications appear here; no email, push, or
								broker orders are sent. A large z-score does not
								establish cointegration or predict a profitable
								trade.
							</p>
						</section>
					</div>
				</div>
				<section className={panel}>
					<div className="mb-5 flex items-center justify-between">
						<h2 className="flex items-center gap-2 font-semibold">
							<Bell size={18} className="text-emerald-400" />
							Your pair alerts
						</h2>
						<span className="text-xs text-slate-500">
							{alerts.filter((a) => a.active).length} active
						</span>
					</div>
					{!alerts.length ? (
						<p className="py-6 text-center text-sm text-slate-500">
							No pair alerts yet. Preview a pair and create your
							first rule.
						</p>
					) : (
						<ul className="divide-y divide-[#243141]">
							{alerts.map((a) => (
								<li key={a.id} className="py-4">
									<div className="flex flex-wrap items-center justify-between gap-4">
										<div>
											<h3 className="font-semibold">
												{a.ticker_a} / {a.ticker_b}
											</h3>
											<p className="mt-1 text-xs text-slate-400">
												{conditionLabel(a)} ·{" "}
												{a.repeat
													? "Repeating"
													: "One-shot"}
												{a.latest
													? ` · Last value ${number(a.latest.value)} (${a.latest.as_of})`
													: ""}
											</p>
										</div>
										<div className="flex items-center gap-3">
											<span
												className={`rounded-full px-2 py-1 text-[11px] ${a.last_error ? "bg-red-400/10 text-red-300" : a.active ? "bg-blue-400/10 text-blue-300" : "bg-slate-400/10 text-slate-300"}`}
											>
												{a.last_error
													? "Data unavailable"
													: a.active
														? "Watching"
														: a.triggered_at
															? "Triggered"
															: "Paused"}
											</span>
											<button
												className="text-xs text-slate-400 hover:text-white"
												disabled={busy}
												onClick={() => edit(a)}
											>
												Edit
											</button>
											<button
												className="p-2 text-slate-400 hover:text-emerald-300"
												disabled={busy}
												aria-label={`${a.active ? "Pause" : "Rearm"} ${a.ticker_a} ${a.ticker_b} pair alert`}
												onClick={() =>
													void mutate(
														() =>
															api(
																`pairs/alerts/${a.id}/state`,
																{
																	method: "POST",
																	body: JSON.stringify(
																		{
																			active: !a.active,
																		},
																	),
																},
															),
														a.active
															? "Pair alert paused."
															: "Pair alert rearmed.",
													)
												}
											>
												{a.active ? (
													<Pause size={15} />
												) : (
													<Play size={15} />
												)}
											</button>
											<button
												className="p-2 text-slate-400 hover:text-red-300"
												disabled={busy}
												aria-label={`Delete ${a.ticker_a} ${a.ticker_b} pair alert`}
												onClick={() =>
													void mutate(
														() =>
															api(
																`pairs/alerts/${a.id}`,
																{
																	method: "DELETE",
																},
															),
														"Pair alert deleted. Prior notifications remain in history.",
													)
												}
											>
												<Trash2 size={15} />
											</button>
										</div>
									</div>
									{a.last_error && (
										<p className="mt-2 text-xs text-red-300">
											{a.last_error}
										</p>
									)}
									<p className="mt-2 text-[11px] text-slate-500">
										{a.last_checked_at
											? `Last checked ${new Date(a.last_checked_at).toLocaleString()}`
											: "Waiting for the first background check"}
									</p>
								</li>
							))}
						</ul>
					)}
				</section>
				<section className={panel}>
					<h2 className="mb-4 font-semibold">
						Recent pair notifications
					</h2>
					{!events.length ? (
						<p className="py-4 text-sm text-slate-500">
							Triggered alerts will appear here and persist across
							restarts.
						</p>
					) : (
						<ul className="max-h-80 divide-y divide-[#243141] overflow-auto">
							{events.map((event) => (
								<li
									key={event.id}
									className="flex flex-wrap justify-between gap-3 py-4"
								>
									<div>
										<b className="text-sm">
											{event.config.ticker_a} /{" "}
											{event.config.ticker_b}
										</b>
										<p className="mt-1 text-xs text-emerald-300">
											{event.snapshot.signal && (
												<span>
													{event.snapshot.signal
														.direction_text ||
														`Long ${event.snapshot.signal.long} / Short ${event.snapshot.signal.short}`}{" "}
													·{" "}
												</span>
											)}
											{conditionLabel(event.config)} ·
											Observed{" "}
											{number(event.snapshot.value)}
										</p>
									</div>
									<div className="text-right text-xs text-slate-400">
										<p>
											Close {event.as_of} ·{" "}
											{event.snapshot.source.toUpperCase()}
										</p>
										<p className="mt-1 text-slate-500">
											{new Date(
												event.created_at,
											).toLocaleString()}
										</p>
									</div>
								</li>
							))}
						</ul>
					)}
				</section>
			</div>
		</main>
	);
}
