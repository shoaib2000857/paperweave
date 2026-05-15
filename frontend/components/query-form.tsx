"use client";

import { FormEvent, useState, useTransition } from "react";

type Props = {
  onSubmit: (question: string, referenceAnswer?: string) => Promise<void>;
};

export function QueryForm({ onSubmit }: Props) {
  const [question, setQuestion] = useState("How did retrieval-augmented methods evolve into graph-based RAG for scientific question answering?");
  const [referenceAnswer, setReferenceAnswer] = useState("");
  const [showReference, setShowReference] = useState(false);
  const [isPending, startTransition] = useTransition();

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    startTransition(async () => {
      await onSubmit(question, referenceAnswer.trim() || undefined);
    });
  };

  return (
    <form onSubmit={handleSubmit} className="rounded-[2rem] border border-white/70 bg-white/80 p-5 shadow-panel backdrop-blur">
      <div className="flex flex-col gap-4 lg:flex-row">
        <div className="flex flex-1 flex-col gap-3">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            className="min-h-28 rounded-[1.5rem] border border-ink/10 bg-paper/80 px-5 py-4 text-sm outline-none transition focus:border-ember"
            placeholder="Ask one scientific-paper question and compare all three pipelines."
          />
          <button
            type="button"
            onClick={() => setShowReference(!showReference)}
            className="self-start text-xs text-ink/45 underline underline-offset-2 transition hover:text-ink/70"
          >
            {showReference ? "Hide reference answer" : "+ Add reference answer (improves evaluation accuracy)"}
          </button>
          {showReference ? (
            <textarea
              value={referenceAnswer}
              onChange={(event) => setReferenceAnswer(event.target.value)}
              className="min-h-16 rounded-[1.5rem] border border-ink/10 bg-paper/80 px-5 py-3 text-xs outline-none transition focus:border-ember/60"
              placeholder="Provide a reference answer — used by BERTScore and the judge LLM for more accurate evaluation."
            />
          ) : null}
        </div>
        <button
          type="submit"
          disabled={isPending}
          className="self-start rounded-[1.5rem] bg-ink px-6 py-4 text-sm font-semibold text-paper transition hover:bg-pine disabled:opacity-60 lg:self-auto"
        >
          {isPending ? "Evaluating..." : "Run All Pipelines"}
        </button>
      </div>
    </form>
  );
}
