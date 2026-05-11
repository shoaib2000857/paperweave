import { SourceRecord } from "@/types/api";

type Props = {
  sources: SourceRecord[];
};

export function GraphPreview({ sources }: Props) {
  return (
    <section className="rounded-[2rem] border border-white/70 bg-white/80 p-6 shadow-panel backdrop-blur">
      <h2 className="mb-4 text-lg font-semibold">Evidence Graph Preview</h2>
      <div className="grid gap-3 md:grid-cols-2">
        {sources.length === 0 ? (
          <div className="rounded-[1.5rem] bg-paper/80 p-4 text-sm text-ink/70">
            Graph-connected evidence will appear here after a GraphRAG run.
          </div>
        ) : (
          sources.slice(0, 6).map((source) => (
            <div key={`${source.id}-${source.score ?? 0}`} className="rounded-[1.5rem] bg-paper/80 p-4">
              <div className="text-xs uppercase tracking-[0.2em] text-ink/45">{source.id}</div>
              <div className="mt-2 text-sm font-semibold">{source.title ?? "Untitled Source"}</div>
              <p className="mt-2 text-sm leading-6 text-ink/80">{source.snippet.slice(0, 180)}</p>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
