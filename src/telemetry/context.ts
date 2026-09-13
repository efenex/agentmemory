// Per-request context propagation. AsyncLocalStorage lets handlers,
// providers, and the logger reach into the same store without threading
// (sessionId, project, traceId, spanId) through every function signature.
//
// Set up at HTTP/MCP entry points after we've identified the session +
// project; read by:
//   - provider spans → attach session/project as attributes
//   - logger (Phase 3) → emit trace_id / span_id alongside JSON fields
//   - metrics labels where the cardinality budget allows it
//
// AsyncLocalStorage propagates across `await` boundaries via the
// node:async_hooks runtime, so we don't need OTel's own context API
// for *our* metadata. (OTel context still handles span parentage.)

import { AsyncLocalStorage } from "node:async_hooks";

export type RequestContext = {
  sessionId?: string;
  project?: string;
  traceId?: string;
  spanId?: string;
};

const storage = new AsyncLocalStorage<RequestContext>();

export function getContext(): RequestContext | undefined {
  return storage.getStore();
}

export function runWithContext<T>(
  ctx: RequestContext,
  fn: () => T | Promise<T>,
): T | Promise<T> {
  return storage.run(ctx, fn);
}

/**
 * Mutate the current context in place. Useful when sessionId/project
 * only become known *after* the outer span starts (e.g. after parsing
 * the request body). No-op when called outside any runWithContext.
 */
export function patchContext(patch: Partial<RequestContext>): void {
  const existing = storage.getStore();
  if (!existing) return;
  Object.assign(existing, patch);
}
