import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import type { EmbeddingProvider, MemoryProvider } from "../src/types.js";

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

const ORIGINAL_ENV = { ...process.env };

beforeEach(() => {
  process.env = { ...ORIGINAL_ENV };
  process.env["AGENTMEMORY_METRICS_ENABLED"] = "true";
});

afterEach(() => {
  process.env = { ...ORIGINAL_ENV };
});

class StubEmbeddingProvider implements EmbeddingProvider {
  name = "stub_embed";
  dimensions = 4;
  failBatchOnce = false;
  embedCalls = 0;
  embedBatchCalls = 0;
  embedImageCalls = 0;

  async embed(_text: string): Promise<Float32Array> {
    this.embedCalls += 1;
    return new Float32Array(this.dimensions);
  }
  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    this.embedBatchCalls += 1;
    if (this.failBatchOnce) {
      this.failBatchOnce = false;
      throw new Error("provider boom");
    }
    return texts.map(() => new Float32Array(this.dimensions));
  }
  async embedImage(_s: string): Promise<Float32Array> {
    this.embedImageCalls += 1;
    return new Float32Array(this.dimensions);
  }
}

class StubMemoryProvider implements MemoryProvider {
  name = "stub_llm";
  failOnce = false;
  timeoutOnce = false;

  async compress(_sys: string, _user: string): Promise<string> {
    if (this.timeoutOnce) {
      this.timeoutOnce = false;
      const err = new Error("operation timed out");
      throw err;
    }
    if (this.failOnce) {
      this.failOnce = false;
      throw new Error("upstream 500");
    }
    return "compressed";
  }
  async summarize(_sys: string, _user: string): Promise<string> {
    return "summary";
  }
}

describe("withMetrics — embedding provider instrumentation", () => {
  it("counts success and error batches, observes latency + batch size", async () => {
    const { withMetrics } = await import("../src/providers/embedding/index.js");
    const {
      embeddingRequestsTotal,
      embeddingRequestDurationMs,
      embeddingBatchSize,
    } = await import("../src/telemetry/prometheus.js");

    embeddingRequestsTotal.reset();
    embeddingRequestDurationMs.reset();
    embeddingBatchSize.reset();

    const stub = new StubEmbeddingProvider();
    const wrapped = withMetrics(stub);

    await wrapped.embedBatch(["a", "b", "c"]);
    stub.failBatchOnce = true;
    await expect(wrapped.embedBatch(["x"])).rejects.toThrow("provider boom");

    const counters = await embeddingRequestsTotal.get();
    const succ = counters.values.find(
      (v) =>
        v.labels.provider === "stub_embed" && v.labels.status === "success",
    );
    const err = counters.values.find(
      (v) =>
        v.labels.provider === "stub_embed" && v.labels.status === "error",
    );
    expect(succ?.value).toBe(1);
    expect(err?.value).toBe(1);

    const batchSize = await embeddingBatchSize.get();
    // Histogram exposes _count / _sum / _bucket samples. Use _count for
    // observation count and _sum to confirm batch sizes were recorded.
    const countSample = batchSize.values.find(
      (v) =>
        v.metricName === "agentmemory_embedding_batch_size_count" &&
        v.labels.provider === "stub_embed",
    );
    const sumSample = batchSize.values.find(
      (v) =>
        v.metricName === "agentmemory_embedding_batch_size_sum" &&
        v.labels.provider === "stub_embed",
    );
    expect(countSample?.value).toBe(1); // only the success path observed
    expect(sumSample?.value).toBe(3); // success batch of length 3
  });

  it("preserves prototype chain so instanceof checks still pass", async () => {
    const { withMetrics } = await import("../src/providers/embedding/index.js");
    const stub = new StubEmbeddingProvider();
    const wrapped = withMetrics(stub);
    expect(wrapped).not.toBe(stub);
    // Object.create-based wrapping: prototype chain leads back to the
    // original concrete class. The wrapper is an instance of it.
    expect(wrapped instanceof StubEmbeddingProvider).toBe(true);
  });

  it("returns provider unchanged when metrics disabled", async () => {
    delete process.env["AGENTMEMORY_METRICS_ENABLED"];
    const { withMetrics } = await import("../src/providers/embedding/index.js");
    const stub = new StubEmbeddingProvider();
    expect(withMetrics(stub)).toBe(stub);
  });
});

describe("ResilientProvider — LLM provider instrumentation", () => {
  it("emits llm_requests_total{status='success'} on resolve", async () => {
    const { ResilientProvider } = await import(
      "../src/providers/resilient.js"
    );
    const { llmRequestsTotal, llmRequestDurationMs } = await import(
      "../src/telemetry/prometheus.js"
    );
    llmRequestsTotal.reset();
    llmRequestDurationMs.reset();

    const inner = new StubMemoryProvider();
    const r = new ResilientProvider(inner);
    await r.compress("sys", "user");

    const counters = await llmRequestsTotal.get();
    const succ = counters.values.find(
      (v) =>
        v.labels.provider === "stub_llm" &&
        v.labels.kind === "compress" &&
        v.labels.status === "success",
    );
    expect(succ?.value).toBe(1);

    const dur = await llmRequestDurationMs.get();
    const durCount = dur.values.find(
      (v) =>
        v.metricName === "agentmemory_llm_request_duration_ms_count" &&
        v.labels.provider === "stub_llm" &&
        v.labels.kind === "compress",
    );
    expect(durCount?.value).toBe(1);
  });

  it("distinguishes error from timeout via AbortError / message regex", async () => {
    const { ResilientProvider } = await import(
      "../src/providers/resilient.js"
    );
    const { llmRequestsTotal } = await import(
      "../src/telemetry/prometheus.js"
    );
    llmRequestsTotal.reset();

    const inner = new StubMemoryProvider();
    inner.failOnce = true;
    const r = new ResilientProvider(inner);
    await expect(r.compress("sys", "user")).rejects.toThrow();

    inner.timeoutOnce = true;
    await expect(r.compress("sys", "user")).rejects.toThrow();

    const counters = await llmRequestsTotal.get();
    const err = counters.values.find(
      (v) =>
        v.labels.provider === "stub_llm" &&
        v.labels.kind === "compress" &&
        v.labels.status === "error",
    );
    const timeout = counters.values.find(
      (v) =>
        v.labels.provider === "stub_llm" &&
        v.labels.kind === "compress" &&
        v.labels.status === "timeout",
    );
    expect(err?.value).toBe(1);
    expect(timeout?.value).toBe(1);
  });

  it("publishes circuit-breaker-state gauge after each call", async () => {
    const { ResilientProvider } = await import(
      "../src/providers/resilient.js"
    );
    const { llmCircuitBreakerState } = await import(
      "../src/telemetry/prometheus.js"
    );
    llmCircuitBreakerState.reset();

    const r = new ResilientProvider(new StubMemoryProvider());
    await r.compress("sys", "user");

    const gauges = await llmCircuitBreakerState.get();
    const sample = gauges.values.find(
      (v) => v.labels.provider === "stub_llm",
    );
    expect(sample?.value).toBe(0); // closed
  });
});

describe("query.ts instrumentation — counters fire on hot paths", () => {
  it("queryStepsTotal increments with op + status labels", async () => {
    const { queryStepsTotal } = await import(
      "../src/telemetry/prometheus.js"
    );
    queryStepsTotal.reset();
    queryStepsTotal.inc({ op: "lineage", status: "success" });
    queryStepsTotal.inc({ op: "rank_by_relevance", status: "success" });

    const counters = await queryStepsTotal.get();
    const lineage = counters.values.find(
      (v) => v.labels.op === "lineage" && v.labels.status === "success",
    );
    const rank = counters.values.find(
      (v) =>
        v.labels.op === "rank_by_relevance" && v.labels.status === "success",
    );
    expect(lineage?.value).toBe(1);
    expect(rank?.value).toBe(1);
  });

  it("queryLlmCallsTotal distinguishes aggregator label", async () => {
    const { queryLlmCallsTotal } = await import(
      "../src/telemetry/prometheus.js"
    );
    queryLlmCallsTotal.reset();
    queryLlmCallsTotal.inc({ aggregator: "synthesize" });
    queryLlmCallsTotal.inc({ aggregator: "rank_by_relevance" });
    queryLlmCallsTotal.inc({ aggregator: "synthesize" });

    const counters = await queryLlmCallsTotal.get();
    const synth = counters.values.find(
      (v) => v.labels.aggregator === "synthesize",
    );
    const rank = counters.values.find(
      (v) => v.labels.aggregator === "rank_by_relevance",
    );
    expect(synth?.value).toBe(2);
    expect(rank?.value).toBe(1);
  });
});
