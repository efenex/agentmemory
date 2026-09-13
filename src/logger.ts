// Thin logging shim for agentmemory.
//
// iii-sdk v0.11 dropped `getContext()`, which had been the source of a
// contextual logger in every function handler (`getContext().logger`).
// Migrating directly to the v0.11 OTEL-based `getLogger()` would force
// every call site to care about the OTEL Logger API shape (`emit(...)`
// with severity numbers and attributes maps). Instead, this module
// exposes a single `logger` singleton with the same `.info/.warn/.error`
// signature the old code used, so the mechanical replacement across
// 30+ function files is: drop the `getContext` import, drop the
// `const ctx = getContext();` line, and rename `ctx.logger.*` to
// `logger.*`. Nothing else changes.
//
// Output goes to stderr as `[agentmemory] <level> <msg> <json-fields>`.
// The iii-engine's `iii-exec` worker runs the agentmemory binary as a
// child process and forwards stderr into `docker logs
// agentmemory-iii-engine-1`, so these lines end up next to the engine's
// own output without needing any OTEL wiring. If we later want
// structured OTEL logs, this file is the only thing that changes.
//
// Phase 3 extension (2026-05-21): when LOKI_PUSH_URL is set, the same
// emit() call additionally queues a structured JSON line for batched
// HTTP push to Loki. Public shape (.info/.warn/.error) is unchanged.
// Loki path is fire-and-forget: a push failure logs a warn locally and
// continues — never crashes the daemon. Trace ID + span ID are pulled
// from the active OTel span context when one exists so Grafana can
// jump log↔trace.
//
// See rohitg00/agentmemory#143 follow-up — the #116 migration updated
// test mocks but left the real `getContext()` imports in place, which
// passed `npm test` (tests mock iii-sdk) and `npm run build` (tsdown
// doesn't type-check) but crashed `node dist/index.mjs` on first
// import.

import { trace as traceApi } from "@opentelemetry/api";

type Fields = Record<string, unknown> | undefined;

function fmt(level: string, msg: string, fields: Fields): string {
  if (!fields || Object.keys(fields).length === 0) {
    return `[agentmemory] ${level} ${msg}`;
  }
  try {
    return `[agentmemory] ${level} ${msg} ${JSON.stringify(fields)}`;
  } catch {
    // Fields contained a circular reference or a BigInt — fall back
    // to the plain message so a log line never throws.
    return `[agentmemory] ${level} ${msg}`;
  }
}

function emit(level: string, msg: string, fields: Fields): void {
  try {
    process.stderr.write(fmt(level, msg, fields) + "\n");
  } catch {
    // stderr is unavailable in some weird test/worker contexts — swallow
    // so no log line can ever crash a handler.
  }
  // Best-effort Loki push. No-op when LOKI_PUSH_URL is unset.
  queueForLoki(level, msg, fields);
}

export const logger = {
  info(msg: string, fields?: Fields): void {
    emit("info", msg, fields);
  },
  warn(msg: string, fields?: Fields): void {
    emit("warn", msg, fields);
  },
  error(msg: string, fields?: Fields): void {
    emit("error", msg, fields);
  },
};

// ---------- Loki push ----------
//
// Pure-fetch HTTP push to Loki's `/loki/api/v1/push`. Batched to keep
// request volume sane (default 100 lines / 2s, whichever comes first).
// LOKI_LEVEL acts as a floor — INFO drops debug, WARN drops debug+info.
// LOKI_PUSH_URL unset → all paths short-circuit; the queue stays empty
// and the timer never starts.

type LokiEntry = [tsNs: string, line: string];

const SEVERITY_ORDER: Record<string, number> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
};

function lokiEnabled(): boolean {
  return !!process.env["LOKI_PUSH_URL"];
}

function lokiLevelFloor(): number {
  const raw = (process.env["LOKI_LEVEL"] || "info").toLowerCase();
  return SEVERITY_ORDER[raw] ?? SEVERITY_ORDER["info"];
}

function lokiBatchSize(): number {
  const raw = process.env["LOKI_BATCH_SIZE"];
  if (!raw) return 100;
  const n = parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : 100;
}

function lokiBatchTimeoutMs(): number {
  const raw = process.env["LOKI_BATCH_TIMEOUT_MS"];
  if (!raw) return 2000;
  const n = parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : 2000;
}

const queue: LokiEntry[] = [];
let flushTimer: ReturnType<typeof setTimeout> | undefined;

function queueForLoki(level: string, msg: string, fields: Fields): void {
  if (!lokiEnabled()) return;
  const severity = SEVERITY_ORDER[level] ?? SEVERITY_ORDER["info"];
  if (severity < lokiLevelFloor()) return;

  // Loki wants nanosecond-precision Unix timestamps as strings.
  const tsNs = (BigInt(Date.now()) * 1000000n).toString();

  // Trace-to-log correlation — Grafana's "log details → trace" link
  // works when these IDs are present on the log line.
  const activeSpan = traceApi.getActiveSpan();
  const traceCtx = activeSpan?.spanContext();

  const payload: Record<string, unknown> = {
    ts: new Date().toISOString(),
    level,
    msg,
    ...(fields || {}),
  };
  if (traceCtx?.traceId) payload.trace_id = traceCtx.traceId;
  if (traceCtx?.spanId) payload.span_id = traceCtx.spanId;

  let line: string;
  try {
    line = JSON.stringify(payload);
  } catch {
    // Fall back to plain msg if fields couldn't be serialized.
    line = JSON.stringify({ ts: payload.ts, level, msg });
  }

  queue.push([tsNs, line]);

  if (queue.length >= lokiBatchSize()) {
    void flush();
    return;
  }
  if (!flushTimer) {
    flushTimer = setTimeout(() => {
      void flush();
    }, lokiBatchTimeoutMs());
    // Don't keep the event loop alive for this timer.
    flushTimer.unref?.();
  }
}

async function flush(): Promise<void> {
  if (flushTimer) {
    clearTimeout(flushTimer);
    flushTimer = undefined;
  }
  if (queue.length === 0) return;
  const url = process.env["LOKI_PUSH_URL"];
  if (!url) {
    queue.length = 0;
    return;
  }

  // Drain. If the push fails the entries are lost — better than
  // unbounded growth or letting failures cascade into the daemon.
  const batch = queue.splice(0, queue.length);
  const body = JSON.stringify({
    streams: [
      {
        stream: {
          service: process.env["OTEL_SERVICE_NAME"] || "agentmemory",
        },
        values: batch,
      },
    ],
  });

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const user = process.env["PROMETHEUS_USER"];
  const pass = process.env["PROMETHEUS_PASSWORD"];
  if (user && pass) {
    headers["Authorization"] =
      "Basic " + Buffer.from(`${user}:${pass}`).toString("base64");
  }

  try {
    const res = await fetch(url, { method: "POST", headers, body });
    if (!res.ok) {
      // Loki uses 204 No Content on success; anything 4xx/5xx means
      // misconfigured auth/labels/URL. Surface to stderr but don't
      // re-queue (we'd just loop on a permanent failure).
      const detail = await res.text().catch(() => "");
      try {
        process.stderr.write(
          `[agentmemory] warn loki: push rejected status=${res.status} detail=${detail.slice(0, 200)}\n`,
        );
      } catch {
        // stderr unavailable — drop the warning.
      }
    }
  } catch (err) {
    try {
      process.stderr.write(
        `[agentmemory] warn loki: push failed ${err instanceof Error ? err.message : String(err)}\n`,
      );
    } catch {
      // stderr unavailable — drop the warning.
    }
  }
}

// Exported for tests; production code doesn't call this.
export async function flushLokiForTests(): Promise<void> {
  await flush();
}

// ---------- boot log ----------
//
// `bootLog` is for the one-shot status lines that every register-*
// function used to dump via `console.log` during engine startup. On a
// fresh install that's ~25 lines of `[agentmemory] X enabled` noise
// before the user can see a prompt. In quiet mode (default), each
// line is captured into a buffer and discarded; the CLI surfaces a
// single compressed summary instead. In verbose mode (set by
// `--verbose` or `AGENTMEMORY_VERBOSE=1`) the lines pass straight
// through to stderr exactly like the old console.log calls.

let bootVerbose =
  process.env["AGENTMEMORY_VERBOSE"] === "1" ||
  process.env["AGENTMEMORY_VERBOSE"] === "true";

const bootBuffer: string[] = [];

export function setBootVerbose(enabled: boolean): void {
  bootVerbose = enabled;
}

export function isBootVerbose(): boolean {
  return bootVerbose;
}

export function bootLog(msg: string): void {
  if (bootVerbose) {
    try {
      process.stderr.write(`[agentmemory] ${msg}\n`);
    } catch {
      // stderr unavailable — drop.
    }
    return;
  }
  if (bootBuffer.length < 500) bootBuffer.push(msg);
}

export function bootWarn(msg: string): void {
  // Warnings always surface; they're rare and the user needs to see
  // them even when the rest of the boot log is suppressed.
  try {
    process.stderr.write(`[agentmemory] warn ${msg}\n`);
  } catch {}
}

export function getBootBuffer(): readonly string[] {
  return bootBuffer;
}
