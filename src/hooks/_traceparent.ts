// Generate a W3C traceparent header for outbound hook→/observe POSTs.
//
// Format (https://www.w3.org/TR/trace-context/#traceparent-header):
//   00-<16-byte trace-id hex>-<8-byte parent-id hex>-<flags hex>
//
// Hooks don't push spans of their own — we just claim a trace ID so
// the server-side api::observe span lands under it. Downstream work
// (mem::observe → mem::compress → mem::summarize → LLM calls) inherits
// the ID through OTel context propagation. Result: one trace per hook
// firing in Tempo. The trace will show a "missing root" in Grafana
// (the hook process generated the ID but emits no span for itself),
// which is intentional — bundling the OTel SDK into the hook .mjs
// files just to push one span isn't worth ~2MB per bundle.

import { randomBytes } from "node:crypto";

export function newTraceparent(): string {
  const traceId = randomBytes(16).toString("hex");
  const spanId = randomBytes(8).toString("hex");
  // flags=01 ⇒ sampled. Backend can sample-down if needed.
  return `00-${traceId}-${spanId}-01`;
}
