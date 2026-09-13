import type { MemoryProvider, CircuitBreakerState } from "../types.js";
import { CircuitBreaker } from "./circuit-breaker.js";
import {
  llmRequestsTotal,
  llmRequestDurationMs,
  llmCircuitBreakerState,
  metricsEnabled,
} from "../telemetry/prometheus.js";
import { withSpan } from "../telemetry/tracer.js";
import { getContext } from "../telemetry/context.js";

const CB_STATE_VALUE: Record<CircuitBreakerState["state"], number> = {
  closed: 0,
  "half-open": 1,
  open: 2,
};

export class ResilientProvider implements MemoryProvider {
  private breaker = new CircuitBreaker();
  name: string;

  constructor(private inner: MemoryProvider) {
    this.name = `resilient(${inner.name})`;
    // Seed the gauge at closed so dashboards have a value before the
    // first call. No-op if metrics are disabled.
    if (metricsEnabled()) {
      llmCircuitBreakerState.set({ provider: this.inner.name }, 0);
    }
  }

  private async call(
    kind: "compress" | "summarize",
    fn: () => Promise<string>,
  ): Promise<string> {
    const ctx = getContext();
    return withSpan(
      `llm.${kind}`,
      {
        // OTel GenAI semantic conventions for cross-tool dashboards.
        "gen_ai.system": this.inner.name,
        "gen_ai.operation.name": kind,
        "agentmemory.session_id": ctx?.sessionId,
        "agentmemory.project": ctx?.project,
      },
      async (span) => {
        if (!this.breaker.isAllowed) {
          span.setAttribute("agentmemory.circuit_state", "open");
          if (metricsEnabled()) {
            llmRequestsTotal.inc({
              provider: this.inner.name,
              kind,
              status: "circuit_open",
            });
          }
          throw new Error("circuit_breaker_open");
        }
        const t0 = Date.now();
        try {
          const result = await fn();
          this.breaker.recordSuccess();
          const dur = Date.now() - t0;
          span.setAttribute("agentmemory.duration_ms", dur);
          if (metricsEnabled()) {
            llmRequestsTotal.inc({
              provider: this.inner.name,
              kind,
              status: "success",
            });
            llmRequestDurationMs.observe(
              { provider: this.inner.name, kind },
              dur,
            );
            this.publishCircuitState();
          }
          return result;
        } catch (err) {
          this.breaker.recordFailure();
          const dur = Date.now() - t0;
          const aborted =
            err instanceof Error &&
            (err.name === "AbortError" || /timed out/i.test(err.message));
          span.setAttribute("agentmemory.duration_ms", dur);
          span.setAttribute(
            "agentmemory.outcome",
            aborted ? "timeout" : "error",
          );
          if (metricsEnabled()) {
            llmRequestsTotal.inc({
              provider: this.inner.name,
              kind,
              status: aborted ? "timeout" : "error",
            });
            llmRequestDurationMs.observe(
              { provider: this.inner.name, kind },
              dur,
            );
            this.publishCircuitState();
          }
          throw err;
        }
      },
    );
  }

  private publishCircuitState(): void {
    llmCircuitBreakerState.set(
      { provider: this.inner.name },
      CB_STATE_VALUE[this.breaker.getState().state] ?? 0,
    );
  }

  async compress(systemPrompt: string, userPrompt: string): Promise<string> {
    return this.call("compress", () =>
      this.inner.compress(systemPrompt, userPrompt),
    );
  }

  async summarize(systemPrompt: string, userPrompt: string): Promise<string> {
    return this.call("summarize", () =>
      this.inner.summarize(systemPrompt, userPrompt),
    );
  }

  get circuitState(): CircuitBreakerState {
    return this.breaker.getState();
  }
}
