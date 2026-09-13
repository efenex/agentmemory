import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

const ORIGINAL_ENV = { ...process.env };

beforeEach(() => {
  // Reset to a clean env so metricsEnabled() defaults to off unless a
  // specific test opts in. Module-level Registry stays loaded between
  // tests; counter resets happen per-test via `metric.reset()`.
  process.env = { ...ORIGINAL_ENV };
  delete process.env["AGENTMEMORY_METRICS_ENABLED"];
});

afterEach(() => {
  process.env = { ...ORIGINAL_ENV };
});

describe("telemetry/prometheus — registry shape", () => {
  it("each defined metric registers and produces text output", async () => {
    process.env["AGENTMEMORY_METRICS_ENABLED"] = "true";
    const mod = await import("../src/telemetry/prometheus.js");
    const { registry, renderMetrics } = mod;

    // Trigger one observation per category so the metric appears in
    // the rendered output (prom-client only emits families with samples).
    mod.toolCallsTotal.inc({ function_id: "mem::test", status: "success" });
    mod.toolLatencyMs.observe({ function_id: "mem::test" }, 42);
    mod.llmRequestsTotal.inc({
      provider: "openai",
      kind: "compress",
      status: "success",
    });
    mod.llmRequestDurationMs.observe(
      { provider: "openai", kind: "compress" },
      123,
    );
    mod.llmCircuitBreakerState.set({ provider: "openai" }, 0);
    mod.embeddingRequestsTotal.inc({ provider: "openai", status: "success" });
    mod.embeddingRequestDurationMs.observe({ provider: "openai" }, 20);
    mod.embeddingBatchSize.observe({ provider: "openai" }, 4);
    mod.hookObservationsTotal.inc({
      hook_type: "post_tool_use",
      project: "test",
    });
    mod.summarizeTotal.inc({ status: "success" });
    mod.summarizeChunksTotal.inc({ status: "success" });
    mod.summarizeChunkDurationMs.observe({ provider: "openai" }, 800);
    mod.queryPipelinesTotal.inc({ result_kind: "synthesis" });
    mod.queryStepsTotal.inc({ op: "lineage", status: "success" });
    mod.queryBudgetUsed.observe(24);
    mod.queryLlmCallsTotal.inc({ aggregator: "synthesize" });
    mod.queryDeadlineExceededTotal.inc();
    mod.observationsTotal.set(496);
    mod.memoriesTotal.set({ type: "decision" }, 12);
    mod.lessonsTotal.set(4574);
    mod.sessionsTotal.set({ status: "active" }, 7);
    mod.graphNodesTotal.set(8300);
    mod.graphEdgesTotal.set(4900);

    const { body, contentType } = await renderMetrics();
    expect(contentType).toContain("text/plain");

    const expectedFamilies = [
      "agentmemory_tool_calls_total",
      "agentmemory_tool_latency_ms",
      "agentmemory_llm_requests_total",
      "agentmemory_llm_request_duration_ms",
      "agentmemory_llm_circuit_breaker_state",
      "agentmemory_embedding_requests_total",
      "agentmemory_embedding_request_duration_ms",
      "agentmemory_embedding_batch_size",
      "agentmemory_hook_observations_total",
      "agentmemory_summarize_total",
      "agentmemory_summarize_chunks_total",
      "agentmemory_summarize_chunk_duration_ms",
      "agentmemory_query_pipelines_total",
      "agentmemory_query_steps_total",
      "agentmemory_query_budget_used",
      "agentmemory_query_llm_calls_total",
      "agentmemory_query_deadline_exceeded_total",
      "agentmemory_observations_total",
      "agentmemory_memories_total",
      "agentmemory_lessons_total",
      "agentmemory_sessions_total",
      "agentmemory_graph_nodes_total",
      "agentmemory_graph_edges_total",
    ];
    for (const family of expectedFamilies) {
      expect(body, `missing metric family ${family}`).toContain(family);
    }
    // Default Node.js process metrics are present too.
    expect(body).toContain("process_cpu_user_seconds_total");
    expect(body).toContain("nodejs_eventloop_lag_seconds");

    // Sanity-check Prometheus text format markers.
    expect(body).toContain("# HELP ");
    expect(body).toContain("# TYPE ");

    // Cleanup so subsequent tests don't see these samples.
    mod.toolCallsTotal.reset();
    mod.toolLatencyMs.reset();
    mod.llmRequestsTotal.reset();
    mod.llmRequestDurationMs.reset();
    mod.llmCircuitBreakerState.reset();
    mod.embeddingRequestsTotal.reset();
    mod.embeddingRequestDurationMs.reset();
    mod.embeddingBatchSize.reset();
    mod.hookObservationsTotal.reset();
    mod.summarizeTotal.reset();
    mod.summarizeChunksTotal.reset();
    mod.summarizeChunkDurationMs.reset();
    mod.queryPipelinesTotal.reset();
    mod.queryStepsTotal.reset();
    mod.queryBudgetUsed.reset();
    mod.queryLlmCallsTotal.reset();
    mod.queryDeadlineExceededTotal.reset();
    mod.observationsTotal.reset();
    mod.memoriesTotal.reset();
    mod.lessonsTotal.reset();
    mod.sessionsTotal.reset();
    mod.graphNodesTotal.reset();
    mod.graphEdgesTotal.reset();

    expect(registry).toBeDefined();
  });
});

describe("telemetry/prometheus — instrumentedTrigger / instrumentedDispatch", () => {
  it("instrumentedTrigger increments success on resolve, error on throw", async () => {
    process.env["AGENTMEMORY_METRICS_ENABLED"] = "true";
    const { instrumentedTrigger, toolCallsTotal, toolLatencyMs } =
      await import("../src/telemetry/prometheus.js");

    toolCallsTotal.reset();
    toolLatencyMs.reset();

    await instrumentedTrigger("mem::test_succeed", async () => "ok");
    await expect(
      instrumentedTrigger("mem::test_fail", async () => {
        throw new Error("boom");
      }),
    ).rejects.toThrow("boom");

    const succ = await toolCallsTotal.get();
    const succSample = succ.values.find(
      (v) =>
        v.labels.function_id === "mem::test_succeed" &&
        v.labels.status === "success",
    );
    const failSample = succ.values.find(
      (v) =>
        v.labels.function_id === "mem::test_fail" &&
        v.labels.status === "error",
    );
    expect(succSample?.value).toBe(1);
    expect(failSample?.value).toBe(1);
  });

  it("instrumentedTrigger short-circuits when metrics disabled", async () => {
    delete process.env["AGENTMEMORY_METRICS_ENABLED"];
    const { instrumentedTrigger, toolCallsTotal } = await import(
      "../src/telemetry/prometheus.js"
    );

    toolCallsTotal.reset();
    await instrumentedTrigger("mem::off", async () => "ok");

    const all = await toolCallsTotal.get();
    const offSample = all.values.find(
      (v) => v.labels.function_id === "mem::off",
    );
    expect(offSample).toBeUndefined();
  });

  it("instrumentedDispatch derives status from status_code", async () => {
    process.env["AGENTMEMORY_METRICS_ENABLED"] = "true";
    const { instrumentedDispatch, toolCallsTotal } = await import(
      "../src/telemetry/prometheus.js"
    );
    toolCallsTotal.reset();

    await instrumentedDispatch("mcp::test_ok", async () => ({
      status_code: 200,
      body: "ok",
    }));
    await instrumentedDispatch("mcp::test_400", async () => ({
      status_code: 400,
      body: { error: "bad" },
    }));
    await instrumentedDispatch("mcp::test_500", async () => ({
      status_code: 500,
      body: { error: "boom" },
    }));

    const all = await toolCallsTotal.get();
    const okSample = all.values.find(
      (v) =>
        v.labels.function_id === "mcp::test_ok" &&
        v.labels.status === "success",
    );
    const badSample = all.values.find(
      (v) =>
        v.labels.function_id === "mcp::test_400" &&
        v.labels.status === "error",
    );
    const boomSample = all.values.find(
      (v) =>
        v.labels.function_id === "mcp::test_500" &&
        v.labels.status === "error",
    );
    expect(okSample?.value).toBe(1);
    expect(badSample?.value).toBe(1);
    expect(boomSample?.value).toBe(1);
  });
});

describe("telemetry/prometheus — metricsEnabled gate", () => {
  it("respects truthy variants of AGENTMEMORY_METRICS_ENABLED", async () => {
    const mod = await import("../src/telemetry/prometheus.js");
    for (const truthy of ["true", "1", "yes", "on", "TRUE", "True"]) {
      process.env["AGENTMEMORY_METRICS_ENABLED"] = truthy;
      expect(mod.metricsEnabled(), truthy).toBe(true);
    }
    for (const falsy of ["", "false", "0", "no", "off"]) {
      process.env["AGENTMEMORY_METRICS_ENABLED"] = falsy;
      expect(mod.metricsEnabled(), falsy).toBe(false);
    }
    delete process.env["AGENTMEMORY_METRICS_ENABLED"];
    expect(mod.metricsEnabled()).toBe(false);
  });
});
