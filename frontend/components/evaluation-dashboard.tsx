"use client";

import { useEffect, useMemo, useState } from "react";

import { AppNav } from "@/components/app-nav";
import { apiUrl, fetchEvaluationDashboard } from "@/lib/evaluation";
import { BenchmarkRecord, EvaluationDashboardResponse, EvaluationLeaderboardRow, PipelineSummary } from "@/types/evaluation";

type SortKey = "weighted" | "latency" | "tokenReduction" | "bertscore" | "judgePass";

const PIPELINE_ORDER = ["llm-only", "basic-rag", "graphrag"] as const;

export function EvaluationDashboard() {
	const [payload, setPayload] = useState<EvaluationDashboardResponse | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(true);
	const [sortKey, setSortKey] = useState<SortKey>("weighted");

	useEffect(() => {
		let active = true;

		function load() {
			fetchEvaluationDashboard()
				.then((response) => {
					if (active) {
						setPayload(response);
						setError(null);
					}
				})
				.catch((fetchError) => {
					if (active) {
						setError(fetchError instanceof Error ? fetchError.message : "Failed to load evaluation dashboard");
					}
				})
				.finally(() => {
					if (active) {
						setLoading(false);
					}
				});
		}

		load();

		const interval = setInterval(load, 12000);

		return () => {
			active = false;
			clearInterval(interval);
		};
	}, []);

	const isLiveData = Boolean(payload?.live);
	const lastQuestion = isLiveData
		? String((payload?.live as Record<string, unknown>)?.question ?? "")
		: "";

	const summaries = useMemo(
		() =>
			Object.entries(payload?.benchmark.summary ?? {}).map(([pipeline, summary]) => ({
				...summary,
				pipeline,
			})),
		[payload],
	);

	const sortedLeaderboard = useMemo(() => {
		const rows = [...(payload?.leaderboard ?? [])];
		rows.sort((left, right) => leaderboardSortValue(right, sortKey) - leaderboardSortValue(left, sortKey));
		return rows;
	}, [payload, sortKey]);

	const questionGroups = useMemo(() => groupQuestions(payload?.benchmark.records ?? []), [payload]);

	const metricCards = useMemo(() => buildMetricCards(sortedLeaderboard, summaries), [sortedLeaderboard, summaries]);
	const chartRows = useMemo(() => buildChartRows(summaries), [summaries]);
	const leaderPipeline = sortedLeaderboard[0]?.pipeline ?? "n/a";
	const bestAccuracy = findBestAccuracy(sortedLeaderboard);
	const fastestPipeline = findFastestPipeline(sortedLeaderboard);
	const highlightedPipeline = sortedLeaderboard[0]?.pipeline ?? "";

	return (
		<main className="min-h-screen bg-slate-950 text-slate-100">
			<div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_top_left,rgba(244,169,92,0.16),transparent_28%),radial-gradient(circle_at_bottom_right,rgba(36,97,92,0.18),transparent_32%),linear-gradient(180deg,rgba(8,15,25,0.96),rgba(2,6,23,1))]" />
			<div className="relative mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-8 px-4 py-6 md:px-6 lg:px-8">
				<header className="rounded-[2.25rem] border border-white/10 bg-white/5 p-6 shadow-[0_30px_100px_rgba(0,0,0,0.28)] backdrop-blur-xl md:p-8">
					<div className="flex flex-col gap-6">
						<div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
							<div className="max-w-3xl space-y-4">
								<AppNav active="evaluation" />
								<div className="flex flex-wrap items-center gap-3">
									<div className="inline-flex rounded-full border border-emerald-400/20 bg-emerald-400/10 px-4 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.32em] text-emerald-200">
										Evaluation observability
									</div>
									{isLiveData ? (
										<div className="inline-flex items-center gap-2 rounded-full border border-sky-400/25 bg-sky-400/10 px-4 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.28em] text-sky-200">
											<span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-400" />
											Live data
										</div>
									) : (
										<div className="inline-flex rounded-full border border-slate-500/30 bg-slate-700/40 px-4 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.28em] text-slate-400">
											Offline data
										</div>
									)}
								</div>
								<div className="space-y-3">
									<h1 className="text-4xl font-semibold tracking-tight text-white md:text-5xl">
										Benchmark intelligence for PaperWeave pipelines
									</h1>
									<p className="max-w-2xl text-sm leading-7 text-slate-300 md:text-base">
										Compare leaderboard performance, token reduction, latency, BERTScore, judge pass rate, and hallucination signals directly inside the app.
									</p>
									{lastQuestion ? (
										<p className="mt-1 max-w-2xl truncate text-xs text-slate-400">
											Last query: <span className="italic text-slate-300">{lastQuestion}</span>
										</p>
									) : null}
								</div>
							</div>
							<div className="grid grid-cols-2 gap-3 text-sm">
								<Stat label="Best pipeline" value={leaderPipeline} />
								<Stat label="Best accuracy" value={bestAccuracy} />
								<Stat label="Fastest pipeline" value={fastestPipeline} />
								<Stat label="Reports" value={String(payload?.report.charts.length ?? 0)} />
							</div>
						</div>

						<div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
							{metricCards.map((card) => (
								<MetricCard key={card.label} {...card} />
							))}
						</div>
					</div>
				</header>

				{loading ? <LoadingState /> : null}
				{error ? <ErrorState message={error} /> : null}

				{!loading && !error && payload ? (
					<>
						<section className="grid gap-6">
							<SectionCard title="Leaderboard" description="Sorted rankings derived from the latest evaluation outputs.">
								<div className="mb-4 flex flex-wrap items-center gap-3">
									<label className="text-xs uppercase tracking-[0.22em] text-slate-400">Sort by</label>
									<select
										value={sortKey}
										onChange={(event) => setSortKey(event.target.value as SortKey)}
										className="rounded-full border border-white/10 bg-slate-900 px-4 py-2 text-sm text-slate-100 outline-none transition focus:border-amber-300/60"
									>
										<option value="weighted">Weighted score</option>
										<option value="latency">Fastest latency</option>
										<option value="tokenReduction">Token reduction</option>
										<option value="bertscore">BERTScore</option>
										<option value="judgePass">Judge pass rate</option>
									</select>
								</div>
								<LeaderboardTable rows={sortedLeaderboard} highlightedPipeline={highlightedPipeline} />
							</SectionCard>
						</section>

						<section className="grid gap-6">
							<SectionCard title="Pipeline signals" description="Comparison charts from the live evaluation summary.">
								<div className="space-y-6">
									<MetricChart title="Token usage" data={chartRows} dataKey="avg_total_tokens" color="#f59e0b" heightClassName="h-[22rem]" />
									<MetricChart title="Latency" data={chartRows} dataKey="avg_total_latency_ms" color="#14b8a6" heightClassName="h-[22rem]" />
									<MetricChart title="BERTScore" data={chartRows} dataKey="avg_bertscore_rescaled_f1" color="#60a5fa" heightClassName="h-[22rem]" />
									<MetricChart title="Judge pass rate" data={chartRows} dataKey="judge_pass_rate" color="#f97316" heightClassName="h-[22rem]" />
									<MetricChart
										title="Hallucination signals"
										data={chartRows}
										dataKey="hallucination_index"
										color="#fb7185"
										secondaryDataKey="avg_fabricated_citation_rate"
										tertiaryDataKey="avg_answer_context_mismatch"
										heightClassName="h-[22rem]"
									/>
								</div>
							</SectionCard>
						</section>

						<section className="grid gap-6">
							<SectionCard
								title="Pipeline answer comparison"
								description="Question-by-question view with answers, retrieved chunks, citations, latency, token counts, and warnings."
							>
								<div className="space-y-6">
									{questionGroups.map((group) => (
										<QuestionComparisonCard key={group.question_id} questionGroup={group} />
									))}
								</div>
							</SectionCard>
						</section>
					</>
				) : null}
			</div>
		</main>
	);
}

function buildMetricCards(rows: EvaluationLeaderboardRow[], summaries: Array<PipelineSummary & { pipeline: string }>) {
	const bestPipeline = rows[0];
	const bestAccuracy = findBestAccuracy(rows);
	const fastestPipeline = findFastestPipeline(rows);
	const avgTokenReduction = average(summaries.map((summary) => summary.avg_token_reduction_pct_vs_llm_only ?? 0));
	const avgJudgePassRate = average(summaries.map((summary) => summary.judge_pass_rate ?? 0));

	return [
		{ label: "Best pipeline", value: bestPipeline?.pipeline ?? "n/a", detail: formatWeighted(bestPipeline) },
		{ label: "Best accuracy", value: bestAccuracy, detail: "BERTScore + judge signal" },
		{ label: "Fastest pipeline", value: fastestPipeline, detail: "Lowest average total latency" },
		{ label: "Avg token reduction", value: `${avgTokenReduction.toFixed(1)}%`, detail: "Across evaluated pipelines" },
		{ label: "Judge pass rate", value: `${avgJudgePassRate.toFixed(1)}%`, detail: "Average pass rate" },
	];
}

function buildChartRows(summaries: Array<PipelineSummary & { pipeline: string }>) {
	return summaries
		.slice()
		.sort(
			(left, right) =>
				PIPELINE_ORDER.indexOf(left.pipeline as typeof PIPELINE_ORDER[number]) -
				PIPELINE_ORDER.indexOf(right.pipeline as typeof PIPELINE_ORDER[number]),
		)
		.map((summary) => ({
			name: summary.pipeline,
			avg_total_tokens: summary.avg_total_tokens ?? 0,
			avg_total_latency_ms: summary.avg_total_latency_ms ?? 0,
			avg_bertscore_rescaled_f1: summary.avg_bertscore_rescaled_f1 ?? 0,
			judge_pass_rate: summary.judge_pass_rate ?? 0,
			avg_fabricated_citation_rate: summary.avg_fabricated_citation_rate ?? 0,
			avg_answer_context_mismatch: summary.avg_answer_context_mismatch ?? 0,
			hallucination_index:
				(summary.avg_fabricated_citation_rate ?? 0) +
				(summary.avg_answer_context_mismatch ?? 0) +
				(summary.judge_hallucination_rate ?? 0),
		}));
}

function groupQuestions(records: BenchmarkRecord[]) {
	const buckets = new Map<string, BenchmarkRecord[]>();
	for (const record of records) {
		const current = buckets.get(record.question_id) ?? [];
		current.push(record);
		buckets.set(record.question_id, current);
	}

	return Array.from(buckets.entries()).map(([questionId, items]) => ({
		question_id: questionId,
		question: items[0]?.question ?? questionId,
		category: items[0]?.category ?? "",
		difficulty: items[0]?.difficulty ?? "",
		records: items.slice().sort((left, right) => PIPELINE_ORDER.indexOf(left.pipeline) - PIPELINE_ORDER.indexOf(right.pipeline)),
	}));
}

function leaderboardSortValue(row: EvaluationLeaderboardRow, sortKey: SortKey): number {
	if (sortKey === "latency") {
		return -(row.avg_total_latency_ms ?? 0);
	}
	if (sortKey === "tokenReduction") {
		return row.avg_token_reduction_pct_vs_llm_only ?? 0;
	}
	if (sortKey === "bertscore") {
		return row.avg_bertscore_rescaled_f1 ?? 0;
	}
	if (sortKey === "judgePass") {
		return row.judge_pass_rate ?? 0;
	}
	return row.hackathon_weighted_score ?? 0;
}

function findBestAccuracy(rows: EvaluationLeaderboardRow[]) {
	const best = rows.slice().sort((left, right) => {
		const leftScore = accuracyScore(left);
		const rightScore = accuracyScore(right);
		return rightScore - leftScore;
	})[0];

	return best ? `${best.pipeline} (${accuracyScore(best).toFixed(2)})` : "n/a";
}

function accuracyScore(row: EvaluationLeaderboardRow) {
	return Math.max(row.avg_bertscore_rescaled_f1 ?? 0, row.judge_pass_rate ?? 0, row.hackathon_answer_accuracy_score ?? 0);
}

function findFastestPipeline(rows: EvaluationLeaderboardRow[]) {
	const best = rows.slice().sort((left, right) => (left.avg_total_latency_ms ?? Number.POSITIVE_INFINITY) - (right.avg_total_latency_ms ?? Number.POSITIVE_INFINITY))[0];
	return best ? `${best.pipeline} (${formatLatency(best.avg_total_latency_ms)})` : "n/a";
}

function formatWeighted(row?: EvaluationLeaderboardRow) {
	return row ? `${(row.hackathon_weighted_score ?? 0).toFixed(2)} weighted` : "n/a";
}

function formatLatency(value?: number) {
	return `${(value ?? 0).toFixed(1)} ms`;
}

function average(values: number[]) {
	return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function MetricCard({ label, value, detail }: { label: string; value: string; detail: string }) {
	return (
		<div className="rounded-[1.75rem] border border-white/10 bg-slate-900/75 p-5 shadow-[0_18px_50px_rgba(0,0,0,0.24)]">
			<div className="text-[0.65rem] uppercase tracking-[0.3em] text-slate-400">{label}</div>
			<div className="mt-3 text-2xl font-semibold text-white">{value}</div>
			<div className="mt-2 text-sm leading-6 text-slate-400">{detail}</div>
		</div>
	);
}

function Stat({ label, value }: { label: string; value: string }) {
	return (
		<div className="rounded-[1.4rem] border border-white/10 bg-slate-900/70 p-4">
			<div className="text-[0.65rem] uppercase tracking-[0.28em] text-slate-400">{label}</div>
			<div className="mt-2 text-base font-semibold text-white">{value}</div>
		</div>
	);
}

function Pill({ children }: { children: React.ReactNode }) {
	return <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">{children}</span>;
}

function LoadingState() {
	return <div className="rounded-[1.5rem] border border-white/10 bg-slate-900/70 p-6 text-sm text-slate-300">Loading evaluation results...</div>;
}

function ErrorState({ message }: { message: string }) {
	return <div className="rounded-[1.5rem] border border-rose-400/20 bg-rose-400/10 p-6 text-sm text-rose-100">{message}</div>;
}

function SectionCard({
	title,
	description,
	children,
}: {
	title: string;
	description: string;
	children: React.ReactNode;
}) {
	return (
		<section className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-[0_24px_80px_rgba(0,0,0,0.22)] backdrop-blur-xl">
			<div className="mb-5 space-y-2">
				<h2 className="text-xl font-semibold text-white">{title}</h2>
				<p className="max-w-3xl text-sm leading-7 text-slate-400">{description}</p>
			</div>
			{children}
		</section>
	);
}

function LeaderboardTable({ rows, highlightedPipeline }: { rows: EvaluationLeaderboardRow[]; highlightedPipeline: string }) {
	return (
		<div className="overflow-x-auto rounded-[1.5rem] border border-white/10 bg-slate-950/70">
			<table className="min-w-full divide-y divide-white/10 text-sm">
				<thead className="bg-white/5 text-left text-xs uppercase tracking-[0.2em] text-slate-400">
					<tr>
						<th className="px-4 py-3">Rank</th>
						<th className="px-4 py-3">Pipeline</th>
						<th className="px-4 py-3">Weighted Score</th>
						<th className="px-4 py-3">Token Reduction</th>
						<th className="px-4 py-3">Latency</th>
						<th className="px-4 py-3">BERTScore</th>
						<th className="px-4 py-3">Judge Pass</th>
					</tr>
				</thead>
				<tbody className="divide-y divide-white/5">
					{rows.map((row) => {
						const isBest = row.pipeline === highlightedPipeline;
						return (
							<tr key={row.pipeline} className={isBest ? "bg-amber-400/8" : "bg-transparent"}>
								<td className="px-4 py-4 text-slate-200">#{row.rank}</td>
								<td className="px-4 py-4">
									<div className="flex flex-wrap items-center gap-2">
										<span className="font-medium text-white">{row.pipeline}</span>
										{isBest ? <span className="rounded-full bg-amber-400/15 px-2 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.24em] text-amber-200">Best</span> : null}
									</div>
								</td>
								<td className="px-4 py-4 text-slate-200">{formatNumber(row.hackathon_weighted_score)}</td>
								<td className="px-4 py-4 text-slate-200">{formatPercent(row.avg_token_reduction_pct_vs_llm_only)}</td>
								<td className="px-4 py-4 text-slate-200">{formatLatency(row.avg_total_latency_ms)}</td>
								<td className="px-4 py-4 text-slate-200">{formatNumber(row.avg_bertscore_rescaled_f1)}</td>
								<td className="px-4 py-4 text-slate-200">{formatPercent(row.judge_pass_rate)}</td>
							</tr>
						);
					})}
				</tbody>
			</table>
		</div>
	);
}

function MetricChart({
	title,
	data,
	dataKey,
	color,
	secondaryDataKey,
	tertiaryDataKey,
	heightClassName = "h-80",
}: {
	title: string;
	data: Array<Record<string, number | string>>;
	dataKey: string;
	color: string;
	secondaryDataKey?: string;
	tertiaryDataKey?: string;
	heightClassName?: string;
}) {
	const values = data.map((item) => Number(item[dataKey] ?? 0));
	const maxValue = Math.max(...values, 1);
	const series = data.map((item, index) => {
		const primary = Number(item[dataKey] ?? 0);
		const secondary = secondaryDataKey ? Number(item[secondaryDataKey] ?? 0) : null;
		const tertiary = tertiaryDataKey ? Number(item[tertiaryDataKey] ?? 0) : null;

		return {
			name: String(item.name ?? `item-${index + 1}`),
			primary,
			secondary,
			tertiary,
			primaryHeight: `${Math.max((primary / maxValue) * 100, 3)}%`,
			secondaryHeight: secondary === null ? null : `${Math.max((secondary / maxValue) * 100, 3)}%`,
			tertiaryHeight: tertiary === null ? null : `${Math.max((tertiary / maxValue) * 100, 3)}%`,
		};
	});

	return (
		<div className="rounded-[1.5rem] border border-white/10 bg-slate-950/70 p-4 shadow-[0_18px_50px_rgba(0,0,0,0.2)]">
			<h3 className="mb-3 text-sm font-semibold text-white">{title}</h3>
			<div className={heightClassName}>
				<div className="flex h-full flex-col rounded-[1.25rem] border border-white/10 bg-slate-950/60 p-3">
					<div className="mb-3 flex flex-wrap gap-3 text-[0.65rem] uppercase tracking-[0.24em] text-slate-400">
						<LegendItem label={title} color={color} />
						{secondaryDataKey ? <LegendItem label={secondaryDataKey} color="#22c55e" /> : null}
						{tertiaryDataKey ? <LegendItem label={tertiaryDataKey} color="#f472b6" /> : null}
					</div>
					<div className="grid flex-1 grid-cols-3 gap-4 md:grid-cols-3">
						{series.map((item) => (
							<div key={item.name} className="flex h-full min-h-0 flex-col justify-end gap-2 rounded-[1rem] bg-white/5 p-3 text-center">
								<div className="flex min-h-0 flex-1 items-end justify-center gap-2">
									{item.secondaryHeight ? (
										<div className="w-3 rounded-t-full" style={{ height: item.secondaryHeight, backgroundColor: "#22c55e" }} />
									) : null}
									<div className="w-6 rounded-t-full" style={{ height: item.primaryHeight, backgroundColor: color }} />
									{item.tertiaryHeight ? (
										<div className="w-3 rounded-t-full" style={{ height: item.tertiaryHeight, backgroundColor: "#f472b6" }} />
									) : null}
								</div>
								<div className="space-y-1">
									<div className="text-xs font-medium text-slate-100">{item.name}</div>
									<div className="text-[0.7rem] text-slate-400">{formatNumber(item.primary)}</div>
								</div>
							</div>
						))}
					</div>
				</div>
			</div>
		</div>
	);
}

function LegendItem({ label, color }: { label: string; color: string }) {
	return (
		<div className="flex items-center gap-2">
			<span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
			<span>{label}</span>
		</div>
	);
}

function QuestionComparisonCard({ questionGroup }: { questionGroup: { question_id: string; question: string; category: string; difficulty: string; records: BenchmarkRecord[] } }) {
	return (
		<div className="rounded-[1.75rem] border border-white/10 bg-slate-950/70 p-5">
			<div className="flex flex-col gap-3 border-b border-white/10 pb-4 md:flex-row md:items-start md:justify-between">
				<div>
					<div className="text-[0.65rem] uppercase tracking-[0.28em] text-slate-400">{questionGroup.category}</div>
					<h3 className="mt-2 text-lg font-semibold text-white">{questionGroup.question}</h3>
					<p className="mt-1 text-sm text-slate-400">Question ID: {questionGroup.question_id}{questionGroup.difficulty ? ` · Difficulty: ${questionGroup.difficulty}` : ""}</p>
				</div>
				<div className="flex flex-wrap gap-2 text-xs text-slate-300">
					<Pill>{questionGroup.records.length} pipeline runs</Pill>
					<Pill>LLM-only / Basic RAG / GraphRAG</Pill>
				</div>
			</div>

			<div className="mt-5 grid gap-4 lg:grid-cols-3">
				{questionGroup.records.map((record) => (
					<PipelineAnswerCard key={`${record.question_id}-${record.pipeline}`} record={record} />
				))}
			</div>
		</div>
	);
}

function PipelineAnswerCard({ record }: { record: BenchmarkRecord }) {
	const warnings = buildWarnings(record);
	return (
		<article className="rounded-[1.5rem] border border-white/10 bg-white/5 p-4">
			<div className="flex items-start justify-between gap-3">
				<div>
					<div className="text-[0.65rem] uppercase tracking-[0.28em] text-slate-400">{record.pipeline}</div>
					<div className="mt-1 text-sm text-slate-300">{formatLatency(record.total_latency_ms)} · {formatTokens(record.total_tokens)}</div>
				</div>
				<div className="rounded-full bg-white/10 px-3 py-1 text-xs text-slate-200">{record.retrieval_mode}</div>
			</div>

			<div className="mt-4 rounded-[1.25rem] border border-white/10 bg-slate-950/60 p-4 text-sm leading-7 text-slate-200">
				{record.answer || "No answer returned for this pipeline run."}
			</div>

			<div className="mt-4 grid grid-cols-2 gap-3 text-xs text-slate-300">
				<MiniMetric label="Latency" value={formatLatency(record.total_latency_ms)} />
				<MiniMetric label="Tokens" value={formatTokens(record.total_tokens)} />
				<MiniMetric label="Citations" value={formatPercent(record.citation_correctness * 100)} />
				<MiniMetric label="Hallucination" value={formatPercent(record.fabricated_citation_rate * 100)} />
			</div>

			{warnings.length > 0 ? (
				<div className="mt-4 rounded-[1.25rem] border border-amber-400/20 bg-amber-400/10 p-3 text-xs leading-6 text-amber-100">
					<div className="mb-1 font-semibold uppercase tracking-[0.22em] text-amber-200">Warnings</div>
					{warnings.map((warning) => (
						<div key={warning}>• {warning}</div>
					))}
				</div>
			) : null}

			<div className="mt-4 space-y-3">
				<div>
					<div className="text-[0.65rem] uppercase tracking-[0.24em] text-slate-500">Retrieved chunks</div>
					<div className="mt-2 space-y-2">
						{record.sources.length > 0 ? (
							record.sources.slice(0, 3).map((source) => (
								<div key={source.id} className="rounded-[1rem] bg-slate-950/70 p-3 text-xs leading-6 text-slate-300">
									<div className="font-medium text-slate-100">{source.title ?? source.id}</div>
									<div className="mt-1 text-slate-400">{source.snippet}</div>
								</div>
							))
						) : (
							<div className="rounded-[1rem] bg-slate-950/70 p-3 text-xs text-slate-400">No retrieved chunks were returned for this pipeline.</div>
						)}
					</div>
				</div>

				<div>
					<div className="text-[0.65rem] uppercase tracking-[0.24em] text-slate-500">Citations</div>
					<div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-300">
						{record.source_titles.length > 0 ? record.source_titles.map((title) => <Pill key={title}>{title}</Pill>) : <Pill>No citations captured</Pill>}
					</div>
				</div>
			</div>
		</article>
	);
}

function MiniMetric({ label, value }: { label: string; value: string }) {
	return (
		<div className="rounded-[1rem] bg-slate-950/70 p-3">
			<div className="text-[0.6rem] uppercase tracking-[0.22em] text-slate-500">{label}</div>
			<div className="mt-1 font-medium text-slate-100">{value}</div>
		</div>
	);
}

function buildWarnings(record: BenchmarkRecord) {
	const warnings = [];
	if (record.error) warnings.push(record.error);
	if (record.fabricated_citation_rate > 0) warnings.push(`Fabricated citation rate is ${record.fabricated_citation_rate.toFixed(2)}.`);
	if (record.answer_context_mismatch > 0) warnings.push(`Answer/context mismatch score is ${record.answer_context_mismatch.toFixed(2)}.`);
	if (record.unsupported_claim_estimate > 0.5) warnings.push(`Unsupported claim estimate is ${record.unsupported_claim_estimate.toFixed(2)}.`);
	return warnings;
}

function formatNumber(value?: number) {
	return value === undefined || Number.isNaN(value) ? "n/a" : value.toFixed(3);
}

function formatPercent(value?: number) {
	return value === undefined || Number.isNaN(value) ? "n/a" : `${value.toFixed(1)}%`;
}

function formatTokens(value?: number) {
	return value === undefined || Number.isNaN(value) ? "n/a" : `${value.toFixed(0)} tokens`;
}
