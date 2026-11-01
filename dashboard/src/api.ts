export type Domain = "code" | "json" | "prose";

export const DOMAINS: Domain[] = ["code", "json", "prose"];

export const DOMAIN_COLORS: Record<Domain, string> = {
  code: "#2563eb",
  json: "#d97706",
  prose: "#059669",
};

export interface RequestMetrics {
  timestamp: number;
  domain: Domain;
  draft_model: string | null;
  k: number;
  acceptance_rate: number | null;
  tokens: number;
  tokens_per_second: number;
  baseline_tokens_per_second: number | null;
  speedup: number | null;
  dispatch_latency_ms: number;
}

export interface DomainStatus {
  domain: Domain;
  draft_model: string;
  k: number;
  speculation_enabled: boolean;
  acceptance_rate: number | null;
  samples: number;
}

export interface Metrics {
  target_model: string;
  baseline_tokens_per_second: number | null;
  domains: DomainStatus[];
  requests: RequestMetrics[];
}

export interface SpeculativeInfo {
  domain: Domain;
  draft_model: string | null;
  k: number;
  accepted_tokens: number;
  drafted_tokens: number;
  acceptance_rate: number;
  cycles: number;
  tokens_per_second: number;
  speedup: number | null;
  dispatch_latency_ms: number;
}

export interface ChatResponse {
  choices: { message: { content: string }; finish_reason: string }[];
  speculative: SpeculativeInfo;
}

export interface ChatOptions {
  temperature: number;
  maxTokens: number;
  json: boolean;
}

export function shortModel(name: string | null): string {
  return name ? name.replace(/^mlx-community\//, "") : "none";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    throw new Error(`${res.status} ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

export function fetchMetrics(limit = 120): Promise<Metrics> {
  return request<Metrics>(`/metrics?limit=${limit}`);
}

export function sendChat(prompt: string, options: ChatOptions): Promise<ChatResponse> {
  return request<ChatResponse>("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: prompt }],
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      ...(options.json ? { response_format: { type: "json_object" } } : {}),
    }),
  });
}
