"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import {
	ArrowLeft,
	ChartCandlestick,
	Code2,
	Download,
	Play,
	Search,
	Upload,
} from "lucide-react";
import { intervals, intervalSeconds } from "@/lib/intervals";
import { api, money } from "@/lib/stockassist";
import RiskComparison from "./risk-comparison";
import EquityChart, { EquityPoint } from "./equity-chart";

type Language = "python" | "cpp";
type Strategy = {
	id?: string;
	template_id?: string;
	description?: string;
	warmup?: number;
	tickers?: string[];
	name: string;
	language: Language;
	code: string;
	params: Record<string, number>;
};
type Candidate = {
	rank: number;
	ticker: string;
	name: string;
	exchange: string;
};
type RunSummary = {
	tickers?: string[];
	id: string;
	name: string;
	language: Language;
	ticker: string;
	status: string;
	created_at: string;
};
type RunInput = Strategy & {
	max_gross_exposure?: number;
	borrow_rate_percent?: number;
	interval?: string;
	ticker: string;
	start_date: string;
	end_date: string;
	initial_cash: number;
	commission: number;
	slippage_bps: number;
	warmup: number;
};
type Result = {
	risk_comparison?: Record<string, Record<string, number | null>>;
	risk_basis?: string;
	benchmark_label?: string;
	open_positions?: Record<string, number>;
	interval?: string;
	metrics: Record<string, number | null>;
	equity_curve: EquityPoint[];
	fills: {
		ticker?: string;
		signal_date: string;
		time: string;
		side: string;
		quantity: number;
		price: number;
		commission: number;
		realized_pnl: number | null;
	}[];
	data: {
		source: string;
		start: string;
		end: string;
		bar_count: number;
		sha256: string;
	};
	assumptions: string[];
	logs: string;
	unfilled_final_signal: number | number[] | null;
};
type Run = {
	id: string;
	status: string;
	request: RunInput;
	result: Result | null;
	error: string | null;
};
const field =
	"w-full rounded-lg border border-[#2b3948] bg-[#0c141e] px-3 py-2.5 text-sm text-slate-100 outline-none focus:border-emerald-400";
const button =
	"rounded-lg bg-emerald-400 px-4 py-2.5 text-sm font-semibold text-[#09251e] hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-40";
const panel = "rounded-2xl border border-[#243141] bg-[#101923] p-5";
const isoDate = (daysAgo: number) =>
	new Date(Date.now() - daysAgo * 86400000).toISOString().slice(0, 10);

export default function ResearchWorkspace() {
	const [presetId, setPresetId] = useState("moving_average");
	const [loadedPreset, setLoadedPreset] = useState<string | null>(
		"moving_average",
	);
	const [templates, setTemplates] = useState<Strategy[]>([]);
	const [saved, setSaved] = useState<Strategy[]>([]);
	const [runs, setRuns] = useState<RunSummary[]>([]);
	const [language, setLanguage] = useState<Language>("python");
	const [code, setCode] = useState("");
	const [drafts, setDrafts] = useState<Partial<Record<Language, string>>>({});
	const [name, setName] = useState("Moving average crossover");
	const [params, setParams] = useState('{"fast": 10, "slow": 30}');
	const [ticker, setTicker] = useState("AAPL");
	const [additionalTickers, setAdditionalTickers] = useState("");
	const [maxGross, setMaxGross] = useState("1");
	const [borrowRate, setBorrowRate] = useState("3");
	const [interval, setIntervalResolution] = useState("1d");
	const [start, setStart] = useState(isoDate(365));
	const [end, setEnd] = useState(isoDate(1));
	const [cash, setCash] = useState("100000");
	const [commission, setCommission] = useState("1");
	const [slippage, setSlippage] = useState("5");
	const [warmup, setWarmup] = useState("30");
	const [activeId, setActiveId] = useState<string | null>(null);
	const [run, setRun] = useState<Run | null>(null);
	const [error, setError] = useState("");
	const [notice, setNotice] = useState("");
	const [forceRerun, setForceRerun] = useState(false);
	const [busy, setBusy] = useState(false);
	const [scanning, setScanning] = useState(false);
	const [scanCode, setScanCode] = useState("HOT_BY_VOLUME");
	const [minPrice, setMinPrice] = useState("5");
	const [maxPrice, setMaxPrice] = useState("1000");
	const [minVolume, setMinVolume] = useState("100000");
	const [candidates, setCandidates] = useState<Candidate[]>([]);
	const [scanSource, setScanSource] = useState("");
	const [scanned, setScanned] = useState(false);
	const [query, setQuery] = useState("");
	const [matches, setMatches] = useState<{ ticker: string; name: string }[]>(
		[],
	);

	useEffect(() => {
		let active = true;
		Promise.all([
			api<Strategy[]>("strategies/examples"),
			api<Strategy[]>("strategies"),
			api<RunSummary[]>("backtests"),
		])
			.then(([examples, strategies, history]) => {
				if (!active) return;
				setTemplates(examples);
				setCode(examples[0].code);
				setSaved(strategies);
				setRuns(history);
			})
			.catch((e) => {
				if (active) setError(e.message);
			});
		const symbol = new URLSearchParams(window.location.search).get(
			"symbol",
		);
		if (symbol && /^[A-Z][A-Z0-9.\-]{0,14}$/.test(symbol))
			setTicker(symbol);
		return () => {
			active = false;
		};
	}, []);

	useEffect(() => {
		if (!activeId) return;
		let alive = true;
		let timer: ReturnType<typeof setTimeout>;
		async function poll() {
			try {
				const latest = await api<Run>(`backtests/${activeId}`);
				if (!alive) return;
				setRun(latest);
				if (latest.status === "queued" || latest.status === "running")
					timer = setTimeout(poll, 1000);
				else setRuns(await api<RunSummary[]>("backtests"));
			} catch (e) {
				if (alive)
					setError(
						e instanceof Error
							? e.message
							: "Could not load backtest.",
					);
			}
		}
		void poll();
		return () => {
			alive = false;
			clearTimeout(timer);
		};
	}, [activeId]);

	useEffect(() => {
		const controller = new AbortController();
		if (!query.trim()) {
			setMatches([]);
			return () => controller.abort();
		}
		const timer = setTimeout(() => {
			api<{ results: { ticker: string; name: string }[] }>(
				`search?ticker=${encodeURIComponent(query)}`,
				{ signal: controller.signal },
			)
				.then((data) => setMatches(data.results))
				.catch((e) => {
					if (!controller.signal.aborted) setError(e.message);
				});
		}, 600);
		return () => {
			clearTimeout(timer);
			controller.abort();
		};
	}, [query]);

	function switchLanguage(next: Language) {
		if (next === language) return;
		setDrafts((previous) => ({ ...previous, [language]: code }));
		setLanguage(next);
		setCode(
			drafts[next] ??
				templates.find(
					(t) =>
						t.language === next && t.template_id === loadedPreset,
				)?.code ??
				"",
		);
	}
	function load(strategy: Strategy) {
		setDrafts((previous) => ({ ...previous, [language]: code }));
		setLoadedPreset(strategy.template_id ?? null);
		setName(strategy.name);
		setLanguage(strategy.language);
		setCode(strategy.code);
		setParams(JSON.stringify(strategy.params, null, 2));
		if (strategy.tickers?.length) {
			setTicker(strategy.tickers[0]);
			setAdditionalTickers(strategy.tickers.slice(1).join(", "));
		} else setAdditionalTickers("");
	}
	function loadPreset() {
		const preset = templates.find(
			(t) => t.template_id === presetId && t.language === language,
		);
		if (!preset) return;
		load(preset);
		setDrafts({});
		setWarmup(String(preset.warmup ?? 0));
		if (preset.tickers?.length) {
			setTicker(preset.tickers[0]);
			setAdditionalTickers(preset.tickers.slice(1).join(", "));
		} else setAdditionalTickers("");
		setNotice(
			`${preset.name} loaded. Code, parameters, and warmup are editable.`,
		);
	}
	function strategyInput(): Strategy {
		const parsed = JSON.parse(params);
		if (
			!parsed ||
			Array.isArray(parsed) ||
			typeof parsed !== "object" ||
			Object.values(parsed).some(
				(value) => typeof value !== "number" || !Number.isFinite(value),
			)
		)
			throw new Error(
				"Parameters must be a JSON object with numeric values.",
			);
		return {
			name,
			language,
			code,
			params: parsed,
			tickers: [
				ticker,
				...additionalTickers.split(/[\s,]+/).filter(Boolean),
			].map((value) => value.toUpperCase().trim()),
		};
	}
	async function save() {
		setError("");
		setBusy(true);
		try {
			await api("strategies", {
				method: "POST",
				body: JSON.stringify(strategyInput()),
			});
			setSaved(await api<Strategy[]>("strategies"));
			setNotice("Strategy version saved.");
		} catch (e) {
			setError(
				e instanceof Error ? e.message : "Could not save strategy.",
			);
		} finally {
			setBusy(false);
		}
	}
	async function changeInterval(value: string) {
		setIntervalResolution(value);
		setError("");
		if (value === "1d") {
			setStart(isoDate(365));
			setEnd(isoDate(1));
			return;
		}
		setStart(new Date(Date.now() - 1800000).toISOString().slice(0, 19));
		setEnd(new Date().toISOString().slice(0, 19));
		setBusy(true);
		try {
			const snapshot = await api<{ bars: { time: string }[] }>(
				`stocks/${encodeURIComponent(ticker)}?interval=${value}`,
			);
			if (snapshot.bars.length) {
				setStart(snapshot.bars[0].time.slice(0, 19));
				setEnd(
					new Date(
						Date.parse(snapshot.bars.at(-1)!.time) +
							intervalSeconds[value] * 1000,
					)
						.toISOString()
						.slice(0, 19),
				);
			}
		} catch (error) {
			setError(
				error instanceof Error
					? error.message
					: "Could not load intraday window.",
			);
		} finally {
			setBusy(false);
		}
	}
	async function execute() {
		setError("");
		setNotice("");
		setBusy(true);
		try {
			const request = {
				...strategyInput(),
				ticker: ticker.toUpperCase().trim(),
				tickers: [
					ticker,
					...additionalTickers.split(/[\s,]+/).filter(Boolean),
				].map((value) => value.toUpperCase().trim()),
				max_gross_exposure: Number(maxGross),
				borrow_rate_percent: Number(borrowRate),
				interval,
				start_date: interval === "1d" ? start : start + "Z",
				end_date: interval === "1d" ? end : end + "Z",
				initial_cash: Number(cash),
				commission: Number(commission),
				slippage_bps: Number(slippage),
				warmup: Number(warmup),
				force_rerun: forceRerun,
			};
			const response = await api<Run & { cache_hit: boolean }>(
				"backtests",
				{
					method: "POST",
					body: JSON.stringify(request),
				},
			);
			setRun(response.status === "completed" ? response : null);
			if (response.cache_hit)
				setNotice(
					"Cached result · identical inputs. Data snapshots refresh after one hour or on a new trading date.",
				);
			setActiveId(response.id);
			setRuns(await api<RunSummary[]>("backtests"));
		} catch (e) {
			setError(
				e instanceof Error ? e.message : "Could not start backtest.",
			);
		} finally {
			setBusy(false);
		}
	}
	async function scan() {
		setError("");
		setScanning(true);
		try {
			const data = await api<{ results: Candidate[]; source: string }>(
				"scanner",
				{
					method: "POST",
					body: JSON.stringify({
						scan_code: scanCode,
						min_price: Number(minPrice),
						max_price: Number(maxPrice),
						min_volume: Number(minVolume),
						limit: 20,
					}),
				},
			);
			setCandidates(data.results);
			setScanSource(data.source);
			setScanned(true);
		} catch (e) {
			setError(e instanceof Error ? e.message : "Scan failed.");
		} finally {
			setScanning(false);
		}
	}
	function download() {
		const url = URL.createObjectURL(
			new Blob([JSON.stringify(run, null, 2)], {
				type: "application/json",
			}),
		);
		const a = document.createElement("a");
		a.href = url;
		a.download = `stockassist-${run?.id}.json`;
		a.click();
		setTimeout(() => URL.revokeObjectURL(url), 1000);
	}
	function loadRun() {
		if (!run) return;
		load(run.request);
		setTicker(run.request.ticker);
		setAdditionalTickers(run.request.tickers?.slice(1).join(", ") ?? "");
		setMaxGross(String(run.request.max_gross_exposure ?? 1));
		setBorrowRate(String(run.request.borrow_rate_percent ?? 3));
		setIntervalResolution(run.request.interval ?? "1d");
		setStart(run.request.start_date.replace(/Z$/, ""));
		setEnd(run.request.end_date.replace(/Z$/, ""));
		setCash(String(run.request.initial_cash));
		setCommission(String(run.request.commission));
		setSlippage(String(run.request.slippage_bps));
		setWarmup(String(run.request.warmup));
	}
	const result = run?.result;
	const running = Boolean(
		activeId && (!run || ["queued", "running"].includes(run.status)),
	);
	const formatPercent = (value: number | null | undefined) =>
		value == null ? "—" : `${value.toFixed(2)}%`;

	return (
		<main className="min-h-screen bg-[#0a111a] text-slate-200">
			<header className="border-b border-[#233040] bg-[#0d1620]">
				<div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-4 px-5 py-5 lg:px-10">
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
							aria-current="page"
							className="text-emerald-400"
						>
							Research & backtest
						</Link>
						<Link
							href="/pairs"
							className="text-slate-400 hover:text-white"
						>
							Pair alerts
						</Link>
					</nav>
				</div>
			</header>
			<div className="mx-auto max-w-[1600px] space-y-6 px-5 py-8 lg:px-10">
				<div>
					<p className="mb-2 text-xs uppercase tracking-[.2em] text-emerald-400">
						From an idea to evidence
					</p>
					<h1 className="text-3xl font-semibold tracking-tight">
						Research & backtest
					</h1>
					<p className="mt-2 text-sm text-slate-400">
						Find a candidate. Bring your code. Test it against
						historical prices.
					</p>
				</div>
				{error && (
					<div
						role="alert"
						aria-label="Research error"
						className="rounded-xl border border-red-400/30 bg-red-400/10 p-4 text-sm text-red-200"
					>
						{error}
					</div>
				)}
				{notice && (
					<div role="status" className="text-sm text-emerald-300">
						{notice}
					</div>
				)}
				<div className="grid items-start gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
					<aside className="space-y-6">
						<section className={panel}>
							<h2 className="mb-4 flex items-center gap-2 font-semibold">
								<Search
									size={17}
									className="text-emerald-400"
								/>
								Find stocks
							</h2>
							<form
								className="space-y-3"
								onSubmit={(e) => {
									e.preventDefault();
									void scan();
								}}
							>
								<label className="block text-xs text-slate-400">
									Scan
									<select
										className={`${field} mt-1`}
										value={scanCode}
										onChange={(e) =>
											setScanCode(e.target.value)
										}
									>
										<option value="HOT_BY_VOLUME">
											Active by volume
										</option>
										<option value="TOP_PERC_GAIN">
											Top percentage gainers
										</option>
										<option value="TOP_PERC_LOSE">
											Top percentage losers
										</option>
									</select>
								</label>
								<div className="grid grid-cols-2 gap-2">
									<label className="text-xs text-slate-400">
										Min price
										<input
											className={`${field} mt-1`}
											type="number"
											min="0"
											step="0.01"
											value={minPrice}
											onChange={(e) =>
												setMinPrice(e.target.value)
											}
											required
										/>
									</label>
									<label className="text-xs text-slate-400">
										Max price
										<input
											className={`${field} mt-1`}
											type="number"
											min="0.01"
											step="0.01"
											value={maxPrice}
											onChange={(e) =>
												setMaxPrice(e.target.value)
											}
											required
										/>
									</label>
								</div>
								<label className="block text-xs text-slate-400">
									Minimum volume
									<input
										className={`${field} mt-1`}
										type="number"
										min="0"
										step="1"
										value={minVolume}
										onChange={(e) =>
											setMinVolume(e.target.value)
										}
										required
									/>
								</label>
								<button
									className={`${button} w-full`}
									disabled={scanning}
								>
									{scanning ? "Scanning…" : "Scan US stocks"}
								</button>
							</form>
							<p className="mt-3 text-xs leading-relaxed text-slate-500">
								US major exchanges · USD stocks. Broker-ranked
								candidates, not trade recommendations.
							</p>
							{scanned && (
								<p className="mt-4 text-xs uppercase tracking-wide text-emerald-400">
									{scanSource === "demo"
										? "Synthetic demo scan"
										: "IBKR scan"}{" "}
									· {candidates.length} results
								</p>
							)}
							<ul className="mt-2 max-h-80 divide-y divide-[#243141] overflow-auto">
								{candidates.map((c) => (
									<li key={`${c.rank}-${c.ticker}`}>
										<button
											className={`w-full rounded px-2 py-3 text-left hover:bg-[#203043] ${ticker === c.ticker ? "bg-[#203043]" : ""}`}
											onClick={() => {
												setTicker(c.ticker);
												setNotice(
													`${c.ticker} selected for backtesting.`,
												);
											}}
										>
											<span className="mr-2 text-xs text-slate-500">
												{c.rank}
											</span>
											<b className="text-sm">
												{c.ticker}
											</b>
											<span className="ml-2 text-[10px] text-slate-500">
												{c.exchange}
											</span>
											<span className="mt-1 block truncate text-xs text-slate-400">
												{c.name}
											</span>
										</button>
									</li>
								))}
							</ul>
							{scanned && !candidates.length && (
								<p className="mt-3 text-sm text-slate-400">
									No stocks match these filters.
								</p>
							)}
							<label className="mt-5 block text-xs text-slate-400">
								Search any symbol
								<input
									className={`${field} mt-1`}
									placeholder="Symbol or company"
									value={query}
									onChange={(e) => setQuery(e.target.value)}
								/>
							</label>
							<ul className="max-h-48 overflow-auto">
								{matches.map((m) => (
									<li key={m.ticker}>
										<button
											className="w-full rounded p-2 text-left text-xs hover:bg-[#203043]"
											onClick={() => {
												setTicker(m.ticker);
												setQuery("");
											}}
										>
											<b>{m.ticker}</b> · {m.name}
										</button>
									</li>
								))}
							</ul>
						</section>
						<section className={panel}>
							<h2 className="mb-3 font-semibold">
								Saved strategies
							</h2>
							{!saved.length && (
								<p className="text-xs text-slate-500">
									Save a version to reuse it later.
								</p>
							)}
							<ul className="max-h-60 space-y-2 overflow-auto">
								{saved.map((s) => (
									<li key={s.id}>
										<button
											onClick={() => load(s)}
											className="w-full rounded-lg bg-[#182431] px-3 py-2 text-left text-sm"
										>
											{s.name}
											<span className="ml-2 text-[10px] uppercase text-slate-500">
												{s.language}
											</span>
										</button>
									</li>
								))}
							</ul>
						</section>
						<section className={panel}>
							<h2 className="mb-3 font-semibold">Recent runs</h2>
							{!runs.length && (
								<p className="text-xs text-slate-500">
									Your backtests will appear here.
								</p>
							)}
							<ul className="max-h-80 space-y-2 overflow-auto">
								{runs.map((r) => (
									<li key={r.id}>
										<button
											onClick={() => {
												setRun(null);
												setActiveId(r.id);
											}}
											className={`w-full rounded-lg px-3 py-3 text-left text-sm ${activeId === r.id ? "bg-[#294238]" : "bg-[#182431]"}`}
										>
											<b>
												{r.tickers?.join(" / ") ||
													r.ticker}
											</b>{" "}
											· {r.name}
											<span className="mt-1 block text-[11px] text-slate-400">
												{r.language} · {r.status} ·{" "}
												{new Date(
													r.created_at,
												).toLocaleDateString()}
											</span>
										</button>
									</li>
								))}
							</ul>
						</section>
					</aside>
					<div className="min-w-0 space-y-6">
						<section className={panel}>
							<div className="mb-5 flex flex-wrap items-center justify-between gap-3">
								<h2 className="flex items-center gap-2 font-semibold">
									<Code2
										size={18}
										className="text-emerald-400"
									/>
									Strategy lab
								</h2>
								<div className="flex gap-2">
									{(["python", "cpp"] as const).map((l) => (
										<button
											key={l}
											aria-pressed={language === l}
											onClick={() => switchLanguage(l)}
											className={`rounded-lg px-4 py-2 text-xs ${language === l ? "bg-emerald-400/15 text-emerald-300" : "bg-[#1b2938] text-slate-400"}`}
										>
											{l === "python" ? "Python" : "C++"}
										</button>
									))}
								</div>
							</div>
							<div className="mb-5 rounded-xl border border-[#2b3948] bg-[#0c141e] p-4">
								<div className="flex flex-wrap items-end gap-3">
									<label className="min-w-48 flex-1 text-xs text-slate-400">
										Starter strategy
										<select
											className={`${field} mt-1`}
											value={presetId}
											onChange={(e) =>
												setPresetId(e.target.value)
											}
										>
											{templates
												.filter(
													(t) =>
														t.language === "python",
												)
												.map((t) => (
													<option
														key={t.template_id}
														value={t.template_id}
													>
														{t.name}
													</option>
												))}
										</select>
									</label>
									<button
										className="rounded-lg border border-emerald-400/40 px-4 py-2.5 text-sm text-emerald-300 disabled:opacity-40"
										disabled={!templates.length}
										onClick={loadPreset}
									>
										Load preset
									</button>
								</div>
								<p className="mt-3 text-xs leading-relaxed text-slate-400">
									{
										templates.find(
											(t) => t.template_id === presetId,
										)?.description
									}
								</p>
								<p className="mt-2 text-[11px] text-slate-500">
									Loading replaces the editor, parameters, and
									warmup. These are comparison baselines, not
									optimized strategies.
								</p>
							</div>
							<div className="mb-4 flex flex-wrap items-end gap-3">
								<label className="min-w-40 flex-1 text-xs text-slate-400">
									Strategy name
									<input
										className={`${field} mt-1`}
										value={name}
										maxLength={100}
										onChange={(e) =>
											setName(e.target.value)
										}
									/>
								</label>
								<label className="flex cursor-pointer items-center gap-2 rounded-lg border border-[#3c5266] px-3 py-2.5 text-xs">
									<Upload size={14} />
									Import code
									<input
										aria-label="Import strategy file"
										type="file"
										accept=".py,.cpp,.cc,.cxx"
										className="sr-only"
										onChange={async (e) => {
											const file = e.target.files?.[0];
											if (!file) return;
											if (file.size > 64000) {
												setError(
													"Strategy code must be under 64 KB.",
												);
												return;
											}
											const text = await file.text();
											setDrafts((previous) => ({
												...previous,
												[language]: code,
											}));
											setLanguage(
												file.name.endsWith(".py")
													? "python"
													: "cpp",
											);
											setCode(text);
											setName(
												file.name.replace(
													/\.[^.]+$/,
													"",
												),
											);
											e.target.value = "";
										}}
									/>
								</label>
								<button
									className="rounded-lg border border-[#3c5266] px-3 py-2.5 text-xs disabled:opacity-40"
									disabled={busy || !code}
									onClick={() => void save()}
								>
									Save version
								</button>
							</div>
							<textarea
								aria-label="Strategy code"
								spellCheck={false}
								className={`${field} min-h-[360px] resize-y font-mono text-xs leading-6`}
								value={code}
								maxLength={64000}
								onChange={(e) => setCode(e.target.value)}
							/>
							<details className="mt-3 text-xs text-slate-400">
								<summary className="cursor-pointer text-emerald-300">
									Strategy interface & execution
								</summary>
								<div className="mt-3 space-y-2 leading-relaxed">
									<p>
										{language === "python"
											? "Define class Strategy with on_bar(self, ctx). ctx.history is completed OHLCV history, ctx.bar is the latest bar, ctx.portfolio exposes cash/shares/equity, and ctx.params contains your numeric parameters. Return None to hold, or 0–1 as a target invested fraction."
											: 'Include strategy_api.h and export extern "C" double on_bar(const SA_Bar* bars, int count, double cash, int64_t shares, const char* params_json). Return -1 to hold or 0–1 as a target invested fraction. SA_Bar contains timestamp (UTC milliseconds), open, high, low, close, and volume. Use sa_parameter for numeric parameters.'}
									</p>
									<p>
										State persists across bars within one
										run. Signals fill at the next bar’s
										open. Warmup calls build state but do
										not place orders. Use imports or native
										libraries available in the backend
										environment.
									</p>
									<p>
										Run trusted code only. Strategies run
										locally in a separate process with time
										limits; this is not a security sandbox.
									</p>
								</div>
							</details>
							<form
								className="mt-6 space-y-4"
								onSubmit={(e) => {
									e.preventDefault();
									void execute();
								}}
							>
								<label className="block text-xs text-slate-400">
									Backtest interval
									<select
										className={`${field} mt-1`}
										value={interval}
										disabled={busy || running}
										onChange={(e) =>
											void changeInterval(e.target.value)
										}
									>
										{intervals.map((value) => (
											<option key={value} value={value}>
												{value === "1d"
													? "Daily"
													: value}
											</option>
										))}
									</select>
								</label>
								<label className="block text-xs text-slate-400">
									Additional stocks (comma separated, up to 9)
									<input
										className={`${field} mt-1 uppercase`}
										value={additionalTickers}
										onChange={(e) =>
											setAdditionalTickers(e.target.value)
										}
										placeholder="MSFT, NVDA"
									/>
								</label>
								{!!additionalTickers.trim() && (
									<>
										<div className="grid gap-3 sm:grid-cols-2">
											<label className="text-xs text-slate-400">
												Maximum gross target exposure
												<input
													className={field}
													type="number"
													min="0.1"
													max="2"
													step="0.1"
													value={maxGross}
													onChange={(e) =>
														setMaxGross(
															e.target.value,
														)
													}
												/>
											</label>
											<label className="text-xs text-slate-400">
												Annual short borrow rate (%)
												<input
													className={field}
													type="number"
													min="0"
													max="100"
													step="0.1"
													value={borrowRate}
													onChange={(e) =>
														setBorrowRate(
															e.target.value,
														)
													}
												/>
											</label>
										</div>
										<p className="text-xs text-emerald-300">
											Portfolio API: C++ exports
											on_portfolio and writes signed
											target weights; Python returns a
											symbol-to-weight dictionary from
											Strategy.on_bar(ctx), using
											ctx.histories and
											ctx.portfolio.positions. Load a
											portfolio starter above. Positive
											weights are long; negative weights
											are short.
										</p>
									</>
								)}
								<div className="grid gap-3 sm:grid-cols-3">
									<label className="text-xs text-slate-400">
										Ticker
										<input
											className={`${field} mt-1 uppercase`}
											value={ticker}
											required
											pattern="[A-Za-z][A-Za-z0-9.\-]{0,14}"
											onChange={(e) =>
												setTicker(e.target.value)
											}
										/>
									</label>
									<label className="text-xs text-slate-400">
										{interval === "1d"
											? "Start date"
											: "Start time · UTC"}
										<input
											className={`${field} mt-1`}
											type={
												interval === "1d"
													? "date"
													: "datetime-local"
											}
											step={
												interval === "1d"
													? undefined
													: 1
											}
											required
											value={start}
											onChange={(e) =>
												setStart(e.target.value)
											}
										/>
									</label>
									<label className="text-xs text-slate-400">
										{interval === "1d"
											? "End date"
											: "End time · UTC (exclusive)"}
										<input
											className={`${field} mt-1`}
											type={
												interval === "1d"
													? "date"
													: "datetime-local"
											}
											step={
												interval === "1d"
													? undefined
													: 1
											}
											required
											value={end}
											onChange={(e) =>
												setEnd(e.target.value)
											}
										/>
									</label>
								</div>
								<div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
									<label className="text-xs text-slate-400">
										Initial cash · USD
										<input
											className={`${field} mt-1`}
											type="number"
											min="100"
											max="100000000"
											required
											value={cash}
											onChange={(e) =>
												setCash(e.target.value)
											}
										/>
									</label>
									<label className="text-xs text-slate-400">
										Fee per fill · USD
										<input
											className={`${field} mt-1`}
											type="number"
											min="0"
											max="1000"
											step="0.01"
											required
											value={commission}
											onChange={(e) =>
												setCommission(e.target.value)
											}
										/>
									</label>
									<label className="text-xs text-slate-400">
										Slippage · basis points
										<input
											className={`${field} mt-1`}
											type="number"
											min="0"
											max="1000"
											step="0.1"
											required
											value={slippage}
											onChange={(e) =>
												setSlippage(e.target.value)
											}
										/>
									</label>
									<label className="text-xs text-slate-400">
										Warmup bars
										<input
											className={`${field} mt-1`}
											type="number"
											min="0"
											max="250"
											required
											value={warmup}
											onChange={(e) =>
												setWarmup(e.target.value)
											}
										/>
									</label>
								</div>
								<label className="block text-xs text-slate-400">
									Parameters · numeric JSON
									<textarea
										aria-label="Strategy parameters"
										className={`${field} mt-1 h-20 font-mono text-xs`}
										value={params}
										onChange={(e) =>
											setParams(e.target.value)
										}
										required
									/>
								</label>
								<label className="flex items-center gap-2 text-xs text-slate-400">
									<input
										type="checkbox"
										checked={forceRerun}
										onChange={(e) =>
											setForceRerun(e.target.checked)
										}
									/>
									Force rerun (skip result cache)
								</label>
								<div className="flex flex-wrap items-center justify-between gap-3">
									<p className="max-w-lg text-xs leading-relaxed text-slate-500">
										Uses completed regular-session bars: up
										to 500 daily bars, or 50,000 intraday
										bars across 40 historical pages.
										Intraday times are UTC; small bars have
										limited historical availability. No
										brokerage orders are sent.
									</p>
									<button
										className={`${button} flex items-center gap-2`}
										disabled={busy || running || !code}
									>
										<Play size={15} />
										{running
											? "Backtest running…"
											: "Run backtest"}
									</button>
								</div>
							</form>
						</section>
						{running && (
							<div
								role="status"
								className={`${panel} text-sm text-emerald-300`}
							>
								Loading historical data and executing strategy…
								You can review other runs while this finishes.
							</div>
						)}
						{run?.status === "failed" && (
							<section className={`${panel} border-red-400/30`}>
								<h2 className="mb-3 font-semibold text-red-300">
									Backtest failed
								</h2>
								<pre
									className="max-h-80 overflow-auto whitespace-pre-wrap text-xs text-red-200"
									role="alert"
								>
									{run.error}
								</pre>
							</section>
						)}
						{result && (
							<section className={panel}>
								<div className="mb-5 flex flex-wrap justify-between gap-3">
									<div>
										<h2 className="text-lg font-semibold">
											{run.request.tickers?.join(" / ") ||
												run.request.ticker}{" "}
											· {run.request.name}
										</h2>
										<p className="mt-1 text-xs text-slate-400">
											{run.request.language === "cpp"
												? "C++"
												: "Python"}{" "}
											· {result.data.source.toUpperCase()}{" "}
											· {result.data.start} to{" "}
											{result.data.end} ·{" "}
											{result.data.bar_count}{" "}
											{result.interval ?? "1d"} bars
										</p>
									</div>
									<div className="flex gap-3">
										<button
											onClick={loadRun}
											className="text-xs text-emerald-300"
										>
											Load inputs
										</button>
										<button
											onClick={download}
											className="flex items-center gap-2 text-xs text-emerald-300"
										>
											<Download size={14} />
											Export JSON
										</button>
									</div>
								</div>
								<div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-3">
									{[
										[
											"Final equity",
											money(
												result.metrics.final_equity ??
													0,
											),
										],
										[
											"Total return",
											formatPercent(
												result.metrics
													.total_return_percent,
											),
										],
										[
											"Buy & hold",
											formatPercent(
												result.metrics
													.benchmark_return_percent,
											),
										],
										[
											"Max drawdown",
											formatPercent(
												result.metrics
													.max_drawdown_percent,
											),
										],
										[
											result.interval &&
											result.interval !== "1d"
												? "Sharpe · daily runs only"
												: "Sharpe · 0% risk-free",
											result.metrics.sharpe_ratio?.toFixed(
												2,
											) ?? "—",
										],
										[
											"Fills / fees",
											`${result.metrics.fill_count} / ${money(result.metrics.total_commission ?? 0)}`,
										],
									].map(([label, value]) => (
										<div
											key={label}
											className="rounded-xl bg-[#0b141e] p-4"
										>
											<p className="text-[11px] text-slate-400">
												{label}
											</p>
											<p className="mt-2 text-xl font-semibold">
												{value}
											</p>
										</div>
									))}
								</div>
								<EquityChart
									points={result.equity_curve}
									benchmarkLabel={result.benchmark_label}
								/>
								{result.risk_comparison && (
									<RiskComparison
										rows={result.risk_comparison}
										basis={result.risk_basis}
										benchmarkLabel={result.benchmark_label}
									/>
								)}
								<div className="mt-6">
									<h3 className="mb-3 text-sm font-semibold">
										Simulated fills
									</h3>
									{!result.fills.length ? (
										<p className="py-5 text-sm text-slate-400">
											No fills in this window. Check your
											signal conditions and warmup.
										</p>
									) : (
										<div className="max-h-80 overflow-auto">
											<table className="w-full whitespace-nowrap text-left text-xs">
												<thead className="text-slate-500">
													<tr>
														{[
															"Signal",
															"Fill date",
															"Stock",
															"Side",
															"Shares",
															"Price",
															"Fee",
															"Realized P&L",
														].map((h) => (
															<th
																key={h}
																className="px-2 pb-3 font-normal"
															>
																{h}
															</th>
														))}
													</tr>
												</thead>
												<tbody>
													{result.fills.map(
														(fill, index) => (
															<tr
																key={index}
																className="border-t border-[#243141]"
															>
																<td className="p-2">
																	{
																		fill.signal_date
																	}
																</td>
																<td className="p-2">
																	{fill.time}
																</td>
																<td className="p-2">
																	{fill.ticker ??
																		run
																			?.request
																			.ticker}
																</td>
																<td
																	className={`p-2 uppercase ${fill.side === "buy" ? "text-emerald-300" : "text-orange-300"}`}
																>
																	{fill.side}
																</td>
																<td className="p-2">
																	{
																		fill.quantity
																	}
																</td>
																<td className="p-2">
																	{money(
																		fill.price,
																	)}
																</td>
																<td className="p-2">
																	{money(
																		fill.commission,
																	)}
																</td>
																<td className="p-2">
																	{fill.realized_pnl ==
																	null
																		? "—"
																		: money(
																				fill.realized_pnl,
																			)}
																</td>
															</tr>
														),
													)}
												</tbody>
											</table>
										</div>
									)}
								</div>
								<details className="mt-5 text-xs text-slate-400">
									<summary className="cursor-pointer">
										Assumptions, open positions & logs
									</summary>
									<p className="mt-3">
										{result.open_positions
											? JSON.stringify(
													result.open_positions,
												)
											: result.metrics.open_shares}{" "}
										shares remain open at the final close.
										Sell-fill win rate:{" "}
										{formatPercent(
											result.metrics.win_rate_percent,
										)}
										. Annualized return:{" "}
										{formatPercent(
											result.metrics
												.annualized_return_percent,
										)}
										.
									</p>
									<ul className="mt-3 list-disc space-y-2 pl-4">
										{result.assumptions.map((a) => (
											<li key={a}>{a}</li>
										))}
									</ul>
									<p className="mt-3 break-all">
										Data SHA-256: {result.data.sha256}
									</p>
									{result.logs && (
										<pre className="mt-3 overflow-auto whitespace-pre-wrap">
											{result.logs}
										</pre>
									)}
								</details>
							</section>
						)}
					</div>
				</div>
				<Link
					href="/"
					className="inline-flex items-center gap-2 text-xs text-slate-400"
				>
					<ArrowLeft size={14} />
					Back to market overview
				</Link>
			</div>
		</main>
	);
}
