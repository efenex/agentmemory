import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { trace as traceApi, trace } from "@opentelemetry/api";

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

// Stand up an in-memory exporter as the *global* tracer provider for
// these tests. Our production code uses its own private provider —
// but the global is the one withSpan() picks up via traceApi.getTracer
// when TEMPO_OTLP_ENDPOINT is unset (since initTracer() short-circuits).
// We register a real provider here so spans are actually recorded.

const ORIGINAL_ENV = { ...process.env };
let exporter: InMemorySpanExporter;
let provider: NodeTracerProvider;

beforeEach(() => {
  process.env = { ...ORIGINAL_ENV };
  delete process.env["TEMPO_OTLP_ENDPOINT"];
  exporter = new InMemorySpanExporter();
  // NodeTracerProvider auto-registers AsyncHooksContextManager so
  // context.with() inside startActiveSpan actually propagates across
  // awaits. BasicTracerProvider does not, which broke parent/child.
  provider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  traceApi.disable();
  provider.register();
});

afterEach(() => {
  process.env = { ...ORIGINAL_ENV };
  traceApi.disable();
  exporter.reset();
});

describe("withSpan", () => {
  it("starts and ends a span around the callback", async () => {
    const { withSpan } = await import("../src/telemetry/tracer.js");
    await withSpan("test.span", { foo: "bar" }, async () => "result");
    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].name).toBe("test.span");
    expect(spans[0].attributes.foo).toBe("bar");
    expect(spans[0].status.code).toBe(1); // OK = 1
  });

  it("records exception + error status on throw", async () => {
    const { withSpan } = await import("../src/telemetry/tracer.js");
    await expect(
      withSpan("test.error", undefined, async () => {
        throw new Error("boom");
      }),
    ).rejects.toThrow("boom");
    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(2); // ERROR = 2
    expect(spans[0].status.message).toBe("boom");
    expect(spans[0].events.some((e) => e.name === "exception")).toBe(true);
  });

  it("nests spans correctly via active context", async () => {
    const { withSpan } = await import("../src/telemetry/tracer.js");
    await withSpan("outer", undefined, async () => {
      await withSpan("inner", undefined, async () => "done");
    });
    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(2);
    // BasicTracerProvider emits child before parent (finish order)
    const outer = spans.find((s) => s.name === "outer")!;
    const inner = spans.find((s) => s.name === "inner")!;
    // Same trace ID means inner was started inside outer's active
    // context. sdk-trace-base 1.30 stores the parent SpanId on the
    // ReadableSpan as a private field that isn't part of the public
    // ReadableSpan type, so we assert the relationship via trace ID
    // (the simpler invariant that's stable across versions).
    expect(inner.spanContext().traceId).toBe(outer.spanContext().traceId);
  });

  it("tracingEnabled reflects TEMPO_OTLP_ENDPOINT presence", async () => {
    const { tracingEnabled } = await import("../src/telemetry/tracer.js");
    expect(tracingEnabled()).toBe(false);
    process.env["TEMPO_OTLP_ENDPOINT"] = "https://tempo.example.test/v1/traces";
    expect(tracingEnabled()).toBe(true);
  });
});

describe("AsyncLocalStorage context", () => {
  it("runWithContext makes ctx visible to nested code", async () => {
    const { runWithContext, getContext } = await import(
      "../src/telemetry/context.js"
    );
    let seen: ReturnType<typeof getContext>;
    await runWithContext({ sessionId: "sess-1", project: "proj-x" }, async () => {
      seen = getContext();
    });
    expect(seen?.sessionId).toBe("sess-1");
    expect(seen?.project).toBe("proj-x");
  });

  it("patchContext mutates the active context", async () => {
    const { runWithContext, getContext, patchContext } = await import(
      "../src/telemetry/context.js"
    );
    await runWithContext({ sessionId: "sess-1" }, async () => {
      patchContext({ project: "added-later" });
      expect(getContext()?.project).toBe("added-later");
    });
  });

  it("getContext returns undefined outside any run", async () => {
    const { getContext } = await import("../src/telemetry/context.js");
    expect(getContext()).toBeUndefined();
  });
});

describe("extractParentContext", () => {
  it("threads an incoming traceparent through withSpan", async () => {
    const { withSpan, extractParentContext, runWithParentContext } =
      await import("../src/telemetry/tracer.js");

    const incomingTraceId = "4bf92f3577b34da6a3ce929d0e0e4736";
    const incomingSpanId = "00f067aa0ba902b7";
    const traceparent = `00-${incomingTraceId}-${incomingSpanId}-01`;

    let capturedTraceId = "";
    await runWithParentContext(
      extractParentContext({ traceparent }),
      async () => {
        await withSpan("child", undefined, async () => {
          capturedTraceId = trace.getActiveSpan()!.spanContext().traceId;
        });
      },
    );

    expect(capturedTraceId).toBe(incomingTraceId);
  });

  it("falls back to a fresh trace when no traceparent header is present", async () => {
    const { withSpan, extractParentContext, runWithParentContext } =
      await import("../src/telemetry/tracer.js");

    let traceId = "";
    await runWithParentContext(
      extractParentContext(undefined),
      async () => {
        await withSpan("orphan", undefined, async () => {
          traceId = trace.getActiveSpan()!.spanContext().traceId;
        });
      },
    );

    expect(traceId).toMatch(/^[0-9a-f]{32}$/);
  });
});
