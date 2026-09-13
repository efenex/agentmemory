// Tempo tracing layer. Independent of iii-sdk's own OTel tracer (which
// pushes to the iii-engine over WebSocket). We register a *local*
// NodeTracerProvider — never call .register() globally — so the global
// tracer keeps belonging to iii-sdk. Spans we explicitly start via
// `tracer.startActiveSpan(...)` end up in Tempo; any other code that
// calls `trace.getTracer(...)` continues to flow through iii-sdk.
//
// IMPORTANT: heavy OTel SDK packages (sdk-trace-node, exporter,
// resources, semantic-conventions) are loaded dynamically inside
// initTracer(). A static top-level import pulls them into the main
// bundle and tsdown reshuffles chunks in a way that creates a TDZ
// ReferenceError at boot (cli.mjs ↔ src-XXX.mjs cycle). Keep the
// top-level imports lightweight — `@opentelemetry/api` is small + has
// no circular footprint with our code.

import {
  trace as traceApi,
  context as contextApi,
  propagation,
  SpanStatusCode,
  type Span,
  type Tracer,
  type SpanOptions,
  type Attributes,
  type Context,
} from "@opentelemetry/api";
import { logger } from "../logger.js";

const TRACER_NAME = "agentmemory";

type AnyTracerProvider = {
  getTracer: (name: string) => Tracer;
  forceFlush?: () => Promise<void>;
  shutdown?: () => Promise<void>;
};

let provider: AnyTracerProvider | null = null;

export function tracingEnabled(): boolean {
  return !!process.env["TEMPO_OTLP_ENDPOINT"];
}

export async function initTracer(): Promise<AnyTracerProvider | null> {
  if (!tracingEnabled()) return null;
  if (provider) return provider;

  const endpoint = process.env["TEMPO_OTLP_ENDPOINT"]!;
  const serviceName = process.env["OTEL_SERVICE_NAME"] || TRACER_NAME;

  const headers: Record<string, string> = {};
  const user = process.env["PROMETHEUS_USER"];
  const pass = process.env["PROMETHEUS_PASSWORD"];
  if (user && pass) {
    headers["Authorization"] =
      "Basic " + Buffer.from(`${user}:${pass}`).toString("base64");
  }

  // Dynamic imports keep these out of the entry-time module graph.
  // See module-level comment for why this matters.
  const [{ NodeTracerProvider }, { BatchSpanProcessor }, otlpHttp, resources, semconv] =
    await Promise.all([
      import("@opentelemetry/sdk-trace-node"),
      import("@opentelemetry/sdk-trace-base"),
      import("@opentelemetry/exporter-trace-otlp-http"),
      import("@opentelemetry/resources"),
      import("@opentelemetry/semantic-conventions"),
    ]);

  const exporter = new otlpHttp.OTLPTraceExporter({
    url: endpoint,
    headers,
  });

  // @opentelemetry/resources 1.30.x exposes the legacy `Resource` class
  // constructor; 2.x renamed to `resourceFromAttributes`. We're pinned
  // to 1.30 to match iii-sdk's OTel deps, so use the class shape.
  const Resource = (resources as { Resource: new (attrs: Record<string, string>) => unknown })
    .Resource;
  const serviceNameAttr =
    (semconv as { ATTR_SERVICE_NAME?: string }).ATTR_SERVICE_NAME ||
    "service.name";

  provider = new NodeTracerProvider({
    resource: new Resource({ [serviceNameAttr]: serviceName }),
    spanProcessors: [new BatchSpanProcessor(exporter)],
  });
  // NOTE: no .register() — that would clobber iii-sdk's global provider.
  logger.info("tempo: tracer initialized", {
    endpoint,
    basicAuth: !!(user && pass),
  });
  return provider;
}

export async function shutdownTracer(): Promise<void> {
  if (!provider) return;
  try {
    await provider.forceFlush?.();
  } catch {
    /* best-effort */
  }
  try {
    await provider.shutdown?.();
  } finally {
    provider = null;
  }
}

export function getTracer(): Tracer {
  if (provider) return provider.getTracer(TRACER_NAME);
  return traceApi.getTracer(TRACER_NAME);
}

/**
 * Start a span, run `fn` inside its active context, record exceptions
 * and final status automatically. Returns whatever `fn` returns.
 *
 * Cheap when no tracer is initialized — the global noop tracer skips
 * the span lifecycle entirely.
 */
export async function withSpan<T>(
  name: string,
  attrs: Attributes | undefined,
  fn: (span: Span) => Promise<T>,
  options: SpanOptions = {},
): Promise<T> {
  const t = getTracer();
  return t.startActiveSpan(
    name,
    { attributes: attrs, ...options },
    async (span) => {
      try {
        const out = await fn(span);
        span.setStatus({ code: SpanStatusCode.OK });
        return out;
      } catch (err) {
        if (err instanceof Error) {
          span.recordException(err);
          span.setStatus({ code: SpanStatusCode.ERROR, message: err.message });
        } else {
          span.setStatus({ code: SpanStatusCode.ERROR });
        }
        throw err;
      } finally {
        span.end();
      }
    },
  );
}

export function spanContextInfo(span: Span): {
  traceId: string;
  spanId: string;
} {
  const sc = span.spanContext();
  return { traceId: sc.traceId, spanId: sc.spanId };
}

/**
 * Extract a W3C traceparent header (and any baggage) from an inbound
 * HTTP carrier, returning a Context with the parent SpanContext
 * attached. Pass the result to `runWithParentContext(ctx, fn)` so
 * `withSpan(...)` inside `fn` creates a child of the upstream trace.
 *
 * iii-sdk's bootstrap already calls `propagation.setGlobalPropagator`
 * with the W3C TraceContextPropagator + W3CBaggagePropagator pair, so
 * this call honors the same wire format any HTTP client uses to send
 * traceparent / tracestate / baggage.
 */
export function extractParentContext(
  headers: Record<string, string | undefined> | undefined,
): Context {
  if (!headers) return contextApi.active();
  // Normalize header keys to lowercase — Node HTTP delivers them that
  // way, but explicit case-insensitive lookup is safer for the iii
  // engine which may forward them in mixed case.
  const carrier: Record<string, string> = {};
  for (const [k, v] of Object.entries(headers)) {
    if (typeof v === "string") carrier[k.toLowerCase()] = v;
  }
  return propagation.extract(contextApi.active(), carrier);
}

/** Run `fn` with the given OTel Context active (for parent propagation). */
export function runWithParentContext<T>(ctx: Context, fn: () => T): T {
  return contextApi.with(ctx, fn);
}

export { contextApi, traceApi };
