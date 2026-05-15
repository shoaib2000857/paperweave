import { AskAllResponse } from "@/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8008";

export async function askAll(question: string, referenceAnswer?: string): Promise<AskAllResponse> {
  const body: Record<string, unknown> = { question };
  if (referenceAnswer?.trim()) {
    body.reference_answer = referenceAnswer.trim();
  }
  const response = await fetch(`${API_BASE}/ask/all`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.json();
}
