// Prometheus metrics exporter (Phase 1 of the local observability layer).
//
// Why prom-client and not iii-sdk's getMeter(): the user runs Prometheus
// in pull mode locally, with no OTel Collector in front of it (per
// docs/plan: "Direct: Prometheus scrapes, Tempo/Loki receive push").
// prom-client is the natural fit for a pull-scrape `/metrics` endpoint;
// iii-sdk's OTel Meter would push via OTLP, which would require a
// receiver this setup doesn't have. Phase 2 (Tempo traces) uses the
// OTel SDK directly because Tempo IS an OTLP receiver — different
// transport, different library.
//
// Gated by AGENTMEMORY_METRICS_ENABLED. Default off — keeps the
// runtime impact zero for any operator who hasn't opted in.

import * as prom from "prom-client";
import type { MetricsStore } from "../eval/metrics-store.js";
import type { StateKV } from "../state/kv.js";
import { KV } from "../state/schema.js";
import type {
  Memory,
  Session,
  Lesson,
  GraphNode,
  GraphEdge,
} from "../types.js";
import { logger } from "../logger.js";

const PREFIX = "agentmemory_";

export const registry = new prom.Registry();

/**
 * True when AGENTMEMORY_METRICS_ENABLED is set to a truthy string.
 * Single source of truth — every instrumented site short-circuits on this.
 */
export function metricsEnabled(): boolean {
  const v = (process.env["AGENTMEMORY_METRICS_ENABLED"] || "").toLowerCase();
  return v === "true" || v === "1" || v === "yes" || v === "on";
}

// ---------------------------------------------------------------------------
// Metric definitions
// ---------------------------------------------------------------------------

// Histogram buckets tuned for the observed latency ranges in this codebase:
//   - tool calls: most under 100ms, p99 around 1-5s (search, smart-search)
//   - LLM calls: 500ms (cached) to 120s (timeout); want fine resolution
//     across that range
//   - embedding: 10ms (cached) to 5s
const TOOL_LATENCY_BUCKETS = [
  5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000,
];
const LLM_LATENCY_BUCKETS = [
  100, 500, 1000, 2500, 5000, 10000, 20000, 30000, 60000, 90000, 120000,
];
const EMBEDDING_LATENCY_BUCKETS = [
  5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000,
];

export const toolCallsTotal = new prom.Counter({
  name: `${PREFIX}tool_calls_total`,
  help: "Total handler / SDK function calls dispatched at the daemon entry point.",
  labelNames: ["function_id", "status"] as const,
  registers: [registry],
});

export const toolLatencyMs = new prom.Histogram({
  name: `${PREFIX}tool_latency_ms`,
  help: "Latency of handler / SDK function calls in milliseconds.",
  labelNames: ["function_id"] as const,
  buckets: TOOL_LATENCY_BUCKETS,
  registers: [registry],
});

export const llmRequestsTotal = new prom.Counter({
  name: `${PREFIX}llm_requests_total`,
  help: "Total LLM provider chat-completion requests.",
  labelNames: ["provider", "kind", "status"] as const,
  registers: [registry],
});

export const llmRequestDurationMs = new prom.Histogram({
  name: `${PREFIX}llm_request_duration_ms`,
  help: "Duration of LLM provider chat-completion requests in milliseconds.",
  labelNames: ["provider", "kind"] as const,
  buckets: LLM_LATENCY_BUCKETS,
  registers: [registry],
});

export const llmCircuitBreakerState = new prom.Gauge({
  name: `${PREFIX}llm_circuit_breaker_state`,
  help: "Circuit-breaker state per LLM provider: 0=closed, 1=half-open, 2=open.",
  labelNames: ["provider"] as const,
  registers: [registry],
});

export const embeddingRequestsTotal = new prom.Counter({
  name: `${PREFIX}embedding_requests_total`,
  help: "Total embedding provider batch requests.",
  labelNames: ["provider", "status"] as const,
  registers: [registry],
});

export const embeddingRequestDurationMs = new prom.Histogram({
  name: `${PREFIX}embedding_request_duration_ms`,
  help: "Duration of embedding provider batch requests in milliseconds.",
  labelNames: ["provider"] as const,
  buckets: EMBEDDING_LATENCY_BUCKETS,
  registers: [registry],
});

export const embeddingBatchSize = new prom.Histogram({
  name: `${PREFIX}embedding_batch_size`,
  help: "Number of items per embedding batch request.",
  labelNames: ["provider"] as const,
  buckets: [1, 4, 16, 64, 256, 1024, 4096],
  registers: [registry],
});

export const hookObservationsTotal = new prom.Counter({
  name: `${PREFIX}hook_observations_total`,
  help: "Hook observations received via /agentmemory/observe (proxy for hook traffic frequency).",
  labelNames: ["hook_type", "project"] as const,
  registers: [registry],
});

export const summarizeTotal = new prom.Counter({
  name: `${PREFIX}summarize_total`,
  help: "mem::summarize invocations by terminal status.",
  labelNames: ["status"] as const,
  registers: [registry],
});

export const summarizeChunksTotal = new prom.Counter({
  name: `${PREFIX}summarize_chunks_total`,
  help: "Per-chunk LLM call outcomes inside chunked summarize runs.",
  labelNames: ["status"] as const,
  registers: [registry],
});

export const summarizeChunkDurationMs = new prom.Histogram({
  name: `${PREFIX}summarize_chunk_duration_ms`,
  help: "Per-chunk LLM call duration during chunked summarize.",
  labelNames: ["provider"] as const,
  buckets: LLM_LATENCY_BUCKETS,
  registers: [registry],
});

export const queryPipelinesTotal = new prom.Counter({
  name: `${PREFIX}query_pipelines_total`,
  help: "mem::query pipeline invocations by terminal result kind.",
  labelNames: ["result_kind"] as const,
  registers: [registry],
});

export const queryStepsTotal = new prom.Counter({
  name: `${PREFIX}query_steps_total`,
  help: "Per-step execution counts inside mem::query pipelines.",
  labelNames: ["op", "status"] as const,
  registers: [registry],
});

export const queryBudgetUsed = new prom.Histogram({
  name: `${PREFIX}query_budget_used`,
  help: "Cost units consumed per mem::query pipeline (cap 100).",
  buckets: [1, 3, 10, 25, 50, 75, 100],
  registers: [registry],
});

export const queryLlmCallsTotal = new prom.Counter({
  name: `${PREFIX}query_llm_calls_total`,
  help: "LLM aggregator calls inside mem::query pipelines.",
  labelNames: ["aggregator"] as const,
  registers: [registry],
});

export const queryDeadlineExceededTotal = new prom.Counter({
  name: `${PREFIX}query_deadline_exceeded_total`,
  help: "mem::query pipelines that hit the configured deadline (timeoutMs).",
  registers: [registry],
});

// Storage gauges — sampled periodically, not on every write.
export const observationsTotal = new prom.Gauge({
  name: `${PREFIX}observations_total`,
  help: "Total stored observations across all sessions.",
  registers: [registry],
});

export const memoriesTotal = new prom.Gauge({
  name: `${PREFIX}memories_total`,
  help: "Total stored memories (latest revision only).",
  labelNames: ["type"] as const,
  registers: [registry],
});

export const lessonsTotal = new prom.Gauge({
  name: `${PREFIX}lessons_total`,
  help: "Total stored lessons (not deleted).",
  registers: [registry],
});

export const sessionsTotal = new prom.Gauge({
  name: `${PREFIX}sessions_total`,
  help: "Total sessions known to the daemon.",
  labelNames: ["status"] as const,
  registers: [registry],
});

export const graphNodesTotal = new prom.Gauge({
  name: `${PREFIX}graph_nodes_total`,
  help: "Total nodes in the knowledge graph.",
  registers: [registry],
});

export const graphEdgesTotal = new prom.Gauge({
  name: `${PREFIX}graph_edges_total`,
  help: "Total edges in the knowledge graph.",
  registers: [registry],
});

// Node runtime defaults (process_*, nodejs_*).
prom.collectDefaultMetrics({ register: registry });

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Wrap an awaited handler invocation with tool_calls_total + tool_latency_ms.
 * Use at the daemon's entry chokepoints (MCP switch, REST handlers) so each
 * registered function is measured once per call without the function itself
 * having to know about Prometheus.
 *
 * When AGENTMEMORY_METRICS_ENABLED is unset/false, this is a thin
 * pass-through with zero hot-path overhead.
 */
export async function instrumentedTrigger<T>(
  functionId: string,
  fn: () => Promise<T>,
): Promise<T> {
  if (!metricsEnabled()) return fn();
  const t0 = Date.now();
  try {
    const result = await fn();
    toolCallsTotal.inc({ function_id: functionId, status: "success" });
    toolLatencyMs.observe({ function_id: functionId }, Date.now() - t0);
    return result;
  } catch (err) {
    toolCallsTotal.inc({ function_id: functionId, status: "error" });
    toolLatencyMs.observe({ function_id: functionId }, Date.now() - t0);
    throw err;
  }
}

/**
 * Wrap an HTTP-style handler whose result is `{ status_code, body? }`.
 * Success vs error is derived from the status code (4xx/5xx = error,
 * else success) so we don't conflate "the handler returned a 400" with
 * "the handler crashed". A thrown exception is still counted as error.
 *
 * Used at the MCP dispatcher (mcp::tools::call) and REST handler
 * entry points where every case returns a Response without throwing.
 */
export async function instrumentedDispatch<
  T extends { status_code: number },
>(functionId: string, fn: () => Promise<T>): Promise<T> {
  if (!metricsEnabled()) return fn();
  const t0 = Date.now();
  try {
    const result = await fn();
    const status = result.status_code >= 400 ? "error" : "success";
    toolCallsTotal.inc({ function_id: functionId, status });
    toolLatencyMs.observe({ function_id: functionId }, Date.now() - t0);
    return result;
  } catch (err) {
    toolCallsTotal.inc({ function_id: functionId, status: "error" });
    toolLatencyMs.observe({ function_id: functionId }, Date.now() - t0);
    throw err;
  }
}

/**
 * Render the Registry to Prometheus text exposition format.
 * Used by the /agentmemory/metrics HTTP endpoint.
 */
export async function renderMetrics(): Promise<{
  contentType: string;
  body: string;
}> {
  return {
    contentType: registry.contentType,
    body: await registry.metrics(),
  };
}

// ---------------------------------------------------------------------------
// Periodic storage gauge refresh + MetricsStore → Registry sync
// ---------------------------------------------------------------------------

let refreshTimer: ReturnType<typeof setInterval> | undefined;

/**
 * Start a 30s refresh loop that walks KV to populate the storage gauges
 * (observations/memories/lessons/sessions/graph) and syncs the
 * MetricsStore's per-function aggregates into the tool_calls_total
 * counter (so historical KV-stored stats also appear in Prometheus).
 *
 * Idempotent — multiple calls reset the timer.
 */
export function startStorageGaugeRefresh(
  kv: StateKV,
  metricsStore: MetricsStore | undefined,
  intervalMs?: number,
): void {
  if (!metricsEnabled()) return;
  if (refreshTimer) clearInterval(refreshTimer);

  // Default 5 minutes — these gauges are eventually-consistent storage
  // size indicators, not real-time. The earlier 30s default caused the
  // daemon to thrash when a session count grew past a few hundred:
  // observations_total used to iterate *every* session and KV.list
  // its observations namespace, decompressing ~580K records every tick.
  // That's been removed; observations_total now mirrors the incremental
  // hook_observations counter (good enough for "is the daemon receiving
  // data" dashboards).
  const ms =
    intervalMs ??
    Number(process.env["AGENTMEMORY_METRICS_REFRESH_MS"] || "300000");

  const refresh = async (): Promise<void> => {
    try {
      // Sessions: count by status. One KV.list — cheap.
      const sessions = await kv.list<Session>(KV.sessions).catch(() => []);
      const sessionsByStatus: Record<string, number> = {};
      for (const s of sessions) {
        const status =
          (s as Session & { status?: string }).status || "unknown";
        sessionsByStatus[status] = (sessionsByStatus[status] || 0) + 1;
      }
      sessionsTotal.reset();
      for (const [status, n] of Object.entries(sessionsByStatus)) {
        sessionsTotal.set({ status }, n);
      }

      // Memories by type (only isLatest === true entries). One KV.list.
      const memories = await kv.list<Memory>(KV.memories).catch(() => []);
      const memByType: Record<string, number> = {};
      for (const m of memories) {
        if (m.isLatest === false) continue;
        const t = m.type || "fact";
        memByType[t] = (memByType[t] || 0) + 1;
      }
      memoriesTotal.reset();
      for (const [type, n] of Object.entries(memByType)) {
        memoriesTotal.set({ type }, n);
      }

      // Lessons (active, not deleted). One KV.list.
      const lessons = await kv.list<Lesson>(KV.lessons).catch(() => []);
      lessonsTotal.set(lessons.filter((l) => !l.deleted).length);

      // Graph nodes/edges (single KV.list each).
      const nodes = await kv.list<GraphNode>(KV.graphNodes).catch(() => []);
      const edges = await kv.list<GraphEdge>(KV.graphEdges).catch(() => []);
      graphNodesTotal.set(nodes.length);
      graphEdgesTotal.set(edges.length);

      // MetricsStore sync: add the delta of cached per-function totals
      // into the Counter. We track which functions we've already mirrored
      // and only add the delta to avoid double-counting on every refresh.
      if (metricsStore) {
        await syncMetricsStore(metricsStore);
      }
    } catch (err) {
      logger.warn("storage gauge refresh failed", {
        error: err instanceof Error ? err.message : String(err),
      });
    }
  };

  void refresh();
  refreshTimer = setInterval(() => void refresh(), ms);
  refreshTimer.unref?.();
}

// Track the last-synced counts per function so we only add the delta
// to the Counter on each sync (Counter has no .set, only .inc).
const syncedCounts = new Map<
  string,
  { success: number; failure: number }
>();

async function syncMetricsStore(store: MetricsStore): Promise<void> {
  const all = await store.getAll();
  for (const m of all) {
    const prev = syncedCounts.get(m.functionId) || { success: 0, failure: 0 };
    const succDelta = Math.max(0, m.successCount - prev.success);
    const failDelta = Math.max(0, m.failureCount - prev.failure);
    if (succDelta > 0) {
      toolCallsTotal.inc(
        { function_id: m.functionId, status: "success" },
        succDelta,
      );
    }
    if (failDelta > 0) {
      toolCallsTotal.inc(
        { function_id: m.functionId, status: "error" },
        failDelta,
      );
    }
    syncedCounts.set(m.functionId, {
      success: m.successCount,
      failure: m.failureCount,
    });
  }
}

/**
 * Stop the refresh timer. Useful in tests where the timer would
 * otherwise prevent vitest from exiting.
 */
export function stopStorageGaugeRefresh(): void {
  if (refreshTimer) {
    clearInterval(refreshTimer);
    refreshTimer = undefined;
  }
}
