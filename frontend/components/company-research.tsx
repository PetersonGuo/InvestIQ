"use client";
import { useEffect, useState, useRef } from "react";
import { api } from "@/lib/stockassist";
type Metric = {
	label: string;
	value: number | null;
	unit: string;
	basis: string;
	period_start?: string;
	period_end?: string;
	filed?: string;
	filing_url?: string;
};
type Fundamentals = {
	ticker: string;
	name?: string;
	source: string;
	description?: string;
	sic_description?: string;
	total_employees?: number;
	market_cap?: number;
	metrics: Metric[];
	notes: string[];
	ratios?: Record<string, number | string | null>;
};
type News = {
	ticker: string;
	source: string;
	fetched_at: string;
	notes: string[];
	articles: {
		id: string;
		title: string;
		publisher: string;
		published_at: string;
		url: string | null;
		provider?: string;
		article_id?: string;
	}[];
};
const amount = (value: number) =>
	value.toLocaleString("en-US", {
		notation: Math.abs(value) >= 1e6 ? "compact" : "standard",
		maximumFractionDigits: 2,
	});
function safeLink(url: string | null | undefined) {
	return !!url && /^https?:\/\//i.test(url);
}
export default function CompanyResearch({ symbol }: { symbol: string }) {
	const [fundamentals, setFundamentals] = useState<Fundamentals | null>(null);
	const [news, setNews] = useState<News | null>(null);
	const [errors, setErrors] = useState({ fundamentals: "", news: "" });
	const articleRequest = useRef<AbortController | null>(null);
	const [article, setArticle] = useState<{
		title: string;
		text: string;
	} | null>(null);
	const [revision, setRevision] = useState(0);
	useEffect(() => {
		const controller = new AbortController();
		setFundamentals(null);
		setNews(null);
		setArticle(null);
		setErrors({ fundamentals: "", news: "" });
		void api<Fundamentals>(
			`stocks/${encodeURIComponent(symbol)}/fundamentals`,
			{ signal: controller.signal },
		)
			.then((data) => {
				if (!controller.signal.aborted) setFundamentals(data);
			})
			.catch((e) => {
				if (!controller.signal.aborted)
					setErrors((old) => ({ ...old, fundamentals: e.message }));
			});
		void api<News>(`stocks/${encodeURIComponent(symbol)}/news`, {
			signal: controller.signal,
		})
			.then((data) => {
				if (!controller.signal.aborted) setNews(data);
			})
			.catch((e) => {
				if (!controller.signal.aborted)
					setErrors((old) => ({ ...old, news: e.message }));
			});
		return () => {
			controller.abort();
			articleRequest.current?.abort();
		};
	}, [symbol, revision]);
	return (
		<section
			aria-label={`${symbol} company research`}
			className="space-y-6 rounded-2xl border border-[#243141] bg-[#101923] p-6"
		>
			<div className="flex items-center justify-between gap-3">
				<h2 className="font-semibold">Company fundamentals</h2>
				<button
					onClick={() => setRevision((value) => value + 1)}
					className="text-xs text-emerald-300"
				>
					Refresh company research
				</button>
			</div>
			{errors.fundamentals ? (
				<p className="text-sm text-amber-200">{errors.fundamentals}</p>
			) : !fundamentals ? (
				<p className="text-sm text-slate-400">
					Loading {symbol} fundamentals…
				</p>
			) : (
				<>
					<p className="text-xs text-slate-400">
						{fundamentals.name || symbol} ·{" "}
						{fundamentals.source === "demo"
							? "SYNTHETIC DEMO FUNDAMENTALS"
							: "Reported financials · SEC EDGAR"}
					</p>
					{fundamentals.description && (
						<p className="text-sm text-slate-300">
							{fundamentals.description}
						</p>
					)}
					{fundamentals.sic_description && (
						<p className="text-xs text-slate-400">
							{fundamentals.sic_description} ·{" "}
							{fundamentals.total_employees
								? `${amount(fundamentals.total_employees)} employees`
								: "Employee count unavailable"}
						</p>
					)}
					<div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
						{fundamentals.metrics.map((metric) => (
							<div
								key={metric.label}
								className="rounded-lg bg-[#0c141e] p-3"
							>
								<p className="text-xs text-slate-400">
									{metric.label}
								</p>
								<p className="mt-1 text-lg font-semibold">
									{metric.value === null
										? "Unavailable"
										: `${metric.unit.startsWith("USD") ? "$" : ""}${amount(metric.value)}${metric.unit === "USD/shares" ? " / share" : ""}`}
								</p>
								<p className="mt-1 text-xs text-slate-500">
									{metric.basis}
									{metric.period_end
										? ` · through ${metric.period_end}`
										: ""}
								</p>
								{metric.filed && (
									<p className="text-xs text-slate-400">
										{safeLink(metric.filing_url) ? (
											<a
												href={metric.filing_url}
												target="_blank"
												rel="noopener noreferrer"
												className="text-emerald-300"
											>
												Filed {metric.filed} ↗
											</a>
										) : (
											`Filed ${metric.filed}`
										)}
									</p>
								)}
							</div>
						))}
					</div>
					{fundamentals.ratios && (
						<div className="text-sm">
							<p className="mb-2 text-xs text-slate-400">
								Massive valuation ratios ·{" "}
								{String(
									fundamentals.ratios.date ??
										"date unavailable",
								)}{" "}
								· provider TTM basis
							</p>
							<div className="flex flex-wrap gap-5">
								{[
									["P/E", "price_to_earnings"],
									["Price/book", "price_to_book"],
									["Debt/equity", "debt_to_equity"],
									["ROE", "return_on_equity"],
								].map(([label, key]) => (
									<span key={key}>
										{label}:{" "}
										{typeof fundamentals.ratios![key] ===
										"number"
											? amount(
													fundamentals.ratios![
														key
													] as number,
												)
											: "Unavailable"}
									</span>
								))}
							</div>
						</div>
					)}
					{fundamentals.notes.map((note, index) => (
						<p key={index} className="text-xs text-slate-400">
							{note}
						</p>
					))}
				</>
			)}
			<div className="border-t border-[#243141] pt-5">
				<h2 className="mb-3 font-semibold">Company news</h2>
				{errors.news ? (
					<p className="text-sm text-amber-200">{errors.news}</p>
				) : !news ? (
					<p className="text-sm text-slate-400">
						Loading {symbol} news…
					</p>
				) : (
					<>
						<p className="mb-3 text-xs text-slate-400">
							{news.source === "demo"
								? "SYNTHETIC DEMO NEWS"
								: news.source.toUpperCase()}{" "}
							· Checked{" "}
							{new Date(news.fetched_at).toLocaleTimeString()}
						</p>
						{!news.articles.length && (
							<p className="text-sm text-slate-400">
								No headlines available from the connected
								providers.
							</p>
						)}
						<ul className="divide-y divide-[#243141]">
							{news.articles.map((article) => (
								<li key={article.id} className="py-3">
									<p className="text-sm">
										{safeLink(article.url) ? (
											<a
												href={article.url!}
												target="_blank"
												rel="noopener noreferrer"
												className="hover:text-emerald-300"
											>
												{article.title} ↗
											</a>
										) : (
											article.title
										)}
									</p>
									<p className="mt-1 text-xs text-slate-500">
										{article.publisher} ·{" "}
										{article.published_at}
										{news.source === "ibkr"
											? " · headline supplied by TWS"
											: ""}
									</p>
									{article.provider && article.article_id && (
										<button
											className="mt-2 text-xs text-emerald-300"
											onClick={async () => {
												articleRequest.current?.abort();
												const controller =
													new AbortController();
												articleRequest.current =
													controller;
												setArticle({
													title: article.title,
													text: "Loading article…",
												});
												try {
													const result = await api<{
														text: string;
													}>(
														`news/article?provider=${encodeURIComponent(article.provider!)}&article=${encodeURIComponent(article.article_id!)}`,
														{
															signal: controller.signal,
														},
													);
													if (
														!controller.signal
															.aborted
													)
														setArticle({
															title: article.title,
															text: result.text,
														});
												} catch (e) {
													if (
														!controller.signal
															.aborted
													)
														setArticle({
															title: article.title,
															text:
																e instanceof
																Error
																	? e.message
																	: "Article unavailable.",
														});
												}
											}}
										>
											Read article
										</button>
									)}
								</li>
							))}
						</ul>
						{news.notes.map((note, index) => (
							<p
								key={index}
								className="mt-2 text-xs text-amber-200"
							>
								{note}
							</p>
						))}
					</>
				)}
			</div>
			{article && (
				<div className="rounded-xl border border-slate-700 p-4">
					<div className="flex items-start justify-between gap-4">
						<h3 className="text-sm font-semibold">
							{article.title}
						</h3>
						<button
							className="text-xs text-emerald-300"
							onClick={() => {
								articleRequest.current?.abort();
								setArticle(null);
							}}
						>
							Close article
						</button>
					</div>
					<p className="mt-3 whitespace-pre-wrap text-sm text-slate-300">
						{article.text}
					</p>
				</div>
			)}
		</section>
	);
}
