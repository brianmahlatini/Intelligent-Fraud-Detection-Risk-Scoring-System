import type { MetricsResponse, ScoredTransaction, TransactionRequest, TransactionsResponse } from "../types/fraud";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function fetchMetrics() {
  return request<MetricsResponse>("/api/ml/metrics");
}

export function fetchTransactions(limit = 120) {
  return request<TransactionsResponse>(`/api/transactions?limit=${limit}`);
}

export function scoreTransaction(payload: TransactionRequest) {
  return request<ScoredTransaction>("/api/score", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

