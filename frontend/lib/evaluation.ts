import { EvaluationDashboardResponse } from "@/types/evaluation";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8008";

export function apiUrl(path: string): string {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${normalized}`;
}

export async function fetchEvaluationDashboard(): Promise<EvaluationDashboardResponse> {
  const response = await fetch(apiUrl("/evaluation/results"), {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch evaluation dashboard: ${response.status}`);
  }

  return response.json();
}
