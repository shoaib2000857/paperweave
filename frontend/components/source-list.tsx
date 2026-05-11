import { SourceRecord } from "@/types/api";

export function SourceList({ sources }: { sources: SourceRecord[] }) {
  return (
    <section className="rounded-[2rem] border border-white/70 bg-white/80 p-6 shadow-panel backdrop-blur">
      <h2 className="mb-4 text-lg font-semibold">Retrieval Evidence</h2>
      <div className="space-y-3">
        {sources.length === 0 ? (
          <p className="text-sm text-ink/65">No sources returned yet.</p>
        ) : (
          sources.map((source) => (
            <article key={`${source.id}-${source.score ?? 0}`} className="rounded-[1.5rem] bg-paper/80 p-4">
              <div className="flex items-center justify-between gap-3">
                <div className="text-sm font-semibold">{source.title ?? source.id}</div>
                <div className="text-xs text-ink/50">{source.score?.toFixed(3) ?? "n/a"}</div>
              </div>
              <p className="mt-2 text-sm leading-6 text-ink/80">{source.snippet}</p>
            </article>
          ))
        )}
      </div>
    </section>
  );
}
