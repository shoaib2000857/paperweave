import { AskResponse } from "@/types/api";

type Props = {
  title: string;
  result?: AskResponse;
};

export function ResultCard({ title, result }: Props) {
  return (
    <section className="rounded-[2rem] border border-white/70 bg-white/80 p-6 shadow-panel backdrop-blur">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold">{title}</h2>
        <span className="rounded-full bg-ink px-3 py-1 text-xs uppercase tracking-[0.2em] text-paper">
          {result?.pipeline ?? "idle"}
        </span>
      </div>
      <p className="min-h-40 text-sm leading-7 text-ink/85">
        {result?.answer ?? "Run a query to render this pipeline's answer, token usage, timing breakdown, and evidence."}
      </p>
      <div className="mt-6 grid grid-cols-2 gap-3 text-sm">
        <Metric label="Tokens" value={result ? String(result.tokens.total_tokens) : "-"} />
        <Metric label="Latency" value={result ? `${result.latency.toFixed(0)} ms` : "-"} />
        <Metric label="Cost" value={result ? `$${result.estimated_cost.toFixed(5)}` : "-"} />
        <Metric label="Sources" value={result ? String(result.sources.length) : "-"} />
      </div>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[1.25rem] bg-paper/90 p-3">
      <div className="text-xs uppercase tracking-[0.18em] text-ink/45">{label}</div>
      <div className="mt-1 text-base font-semibold">{value}</div>
    </div>
  );
}
