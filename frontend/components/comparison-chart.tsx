"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type Props = {
  data: Array<{
    name: string;
    tokens: number;
    latency: number;
    bertscore: number;
    judge: number;
    hallucination: number;
  }>;
};

export function ComparisonChart({ data }: Props) {
  return (
    <div className="rounded-[2rem] border border-white/70 bg-white/80 p-6 shadow-panel backdrop-blur">
      <h2 className="mb-4 text-lg font-semibold">Live Comparison: Tokens, Latency, Accuracy, Hallucination</h2>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#d6d0c4" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="tokens" fill="#bc5a2b" radius={[8, 8, 0, 0]} />
            <Bar dataKey="latency" fill="#123c38" radius={[8, 8, 0, 0]} />
            <Bar dataKey="bertscore" fill="#2563eb" radius={[8, 8, 0, 0]} />
            <Bar dataKey="judge" fill="#f97316" radius={[8, 8, 0, 0]} />
            <Bar dataKey="hallucination" fill="#ef4444" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
