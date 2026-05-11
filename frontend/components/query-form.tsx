"use client";

import { FormEvent, useState, useTransition } from "react";

type Props = {
  onSubmit: (question: string) => Promise<void>;
};

export function QueryForm({ onSubmit }: Props) {
  const [question, setQuestion] = useState("How did retrieval-augmented methods evolve into graph-based RAG for scientific question answering?");
  const [isPending, startTransition] = useTransition();

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    startTransition(async () => {
      await onSubmit(question);
    });
  };

  return (
    <form onSubmit={handleSubmit} className="rounded-[2rem] border border-white/70 bg-white/80 p-5 shadow-panel backdrop-blur">
      <div className="flex flex-col gap-4 lg:flex-row">
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          className="min-h-28 flex-1 rounded-[1.5rem] border border-ink/10 bg-paper/80 px-5 py-4 text-sm outline-none transition focus:border-ember"
          placeholder="Ask one scientific-paper question and compare all three pipelines."
        />
        <button
          type="submit"
          disabled={isPending}
          className="rounded-[1.5rem] bg-ink px-6 py-4 text-sm font-semibold text-paper transition hover:bg-pine disabled:opacity-60"
        >
          {isPending ? "Running..." : "Run All Pipelines"}
        </button>
      </div>
    </form>
  );
}
