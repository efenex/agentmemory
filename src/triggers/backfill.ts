// One-shot backfill endpoints. Accept pre-built Session /
// CompressedObservation / SessionSummary records and write them
// straight into KV, bypassing the live /observe pipeline's synthetic
// compression + LLM compression paths. Intended for migrating
// external sources (claude-mem, other agent memory tools, manual
// JSON imports) where the upstream already did the structuring work.
//
// Gated by AGENTMEMORY_BACKFILL_ENABLED=true so this endpoint family
// is off in normal operation. Returns 503 when disabled. Every
// imported record gets a marker concept (default "claude-mem-backfill"
// — caller can override via .source field) so the import is
// grep-able and forget-able as a group.
//
// What this does NOT do:
//   - LLM-side compression: caller must supply already-compressed obs
//   - Embedding: indexed into BM25 + vector if providers are loaded,
//     but failures are logged-and-continued (rebuild-graph.sh can
//     reindex the whole corpus after a bulk import)
//   - Auth: piggybacks the standard checkAuth() if AGENTMEMORY_AUTH_SECRET
//     is set in the daemon env

import type { ISdk, ApiRequest } from "iii-sdk";
import type { StateKV } from "../state/kv.js";
import { KV } from "../state/schema.js";
import type {
  Session,
  CompressedObservation,
  SessionSummary,
  ObservationType,
} from "../types.js";
import { timingSafeCompare } from "../auth.js";
import { getSearchIndex, vectorIndexAddGuarded, rebuildVectorMissing } from "../functions/search.js";
import { logger } from "../logger.js";

type Response = {
  status_code: number;
  headers?: Record<string, string>;
  body: unknown;
};

const VALID_OBS_TYPES = new Set<ObservationType>([
  "file_read",
  "file_write",
  "file_edit",
  "command_run",
  "search",
  "web_fetch",
  "conversation",
  "error",
  "decision",
  "discovery",
  "subagent",
  "notification",
  "task",
  "image",
  "other",
]);

function backfillEnabled(): boolean {
  const raw = (process.env["AGENTMEMORY_BACKFILL_ENABLED"] || "").toLowerCase();
  return raw === "true" || raw === "1" || raw === "yes" || raw === "on";
}

function checkAuth(
  req: ApiRequest,
  secret: string | undefined,
): Response | null {
  if (!secret) return null;
  const auth =
    req.headers?.["authorization"] || req.headers?.["Authorization"];
  if (typeof auth !== "string" || !timingSafeCompare(auth, `Bearer ${secret}`)) {
    return { status_code: 401, body: { error: "unauthorized" } };
  }
  return null;
}

function disabledResponse(): Response {
  return {
    status_code: 503,
    body: {
      error: "backfill disabled",
      hint: "set AGENTMEMORY_BACKFILL_ENABLED=true to enable /agentmemory/backfill/* endpoints",
    },
  };
}

function asString(v: unknown): string | undefined {
  return typeof v === "string" && v.length > 0 ? v : undefined;
}

function asStringArray(v: unknown): string[] {
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === "string" && x.length > 0);
}

export function registerBackfillTriggers(sdk: ISdk, kv: StateKV, secret?: string): void {
  // ---------- POST /agentmemory/backfill/session ----------
  sdk.registerFunction(
    "api::backfill::session",
    async (req: ApiRequest): Promise<Response> => {
      if (!backfillEnabled()) return disabledResponse();
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;

      const body = (req.body ?? {}) as Record<string, unknown>;
      const sessionId = asString(body.sessionId);
      const project = asString(body.project);
      if (!sessionId || !project) {
        return {
          status_code: 400,
          body: { error: "sessionId and project are required strings" },
        };
      }
      const cwd = asString(body.cwd) || project;
      const startedAt = asString(body.startedAt) || new Date().toISOString();
      const endedAt = asString(body.endedAt);
      const status =
        body.status === "active" || body.status === "completed" || body.status === "abandoned"
          ? body.status
          : "completed";
      const firstPrompt = asString(body.firstPrompt);
      const summary = asString(body.summary);
      const tags = asStringArray(body.tags);

      const session: Session = {
        id: sessionId,
        project,
        cwd,
        startedAt,
        status,
        observationCount: 0,
        ...(endedAt ? { endedAt } : {}),
        ...(firstPrompt ? { firstPrompt: firstPrompt.slice(0, 200) } : {}),
        ...(summary ? { summary: summary.slice(0, 500) } : {}),
        ...(tags.length ? { tags } : {}),
      };
      await kv.set(KV.sessions, sessionId, session);
      return { status_code: 201, body: { sessionId } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::backfill::session",
    config: { api_path: "/agentmemory/backfill/session", http_method: "POST" },
  });

  // ---------- POST /agentmemory/backfill/observation ----------
  sdk.registerFunction(
    "api::backfill::observation",
    async (req: ApiRequest): Promise<Response> => {
      if (!backfillEnabled()) return disabledResponse();
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;

      const body = (req.body ?? {}) as Record<string, unknown>;
      const id = asString(body.id);
      const sessionId = asString(body.sessionId);
      const title = asString(body.title);
      const timestamp = asString(body.timestamp) || new Date().toISOString();
      if (!id || !sessionId || !title) {
        return {
          status_code: 400,
          body: { error: "id, sessionId, and title are required strings" },
        };
      }
      const rawType = asString(body.type) || "other";
      const type: ObservationType = VALID_OBS_TYPES.has(rawType as ObservationType)
        ? (rawType as ObservationType)
        : "other";
      const importance =
        typeof body.importance === "number" && Number.isFinite(body.importance)
          ? body.importance
          : 5;

      const obs: CompressedObservation = {
        id,
        sessionId,
        timestamp,
        type,
        title: title.slice(0, 200),
        ...(asString(body.subtitle) ? { subtitle: asString(body.subtitle)!.slice(0, 200) } : {}),
        facts: asStringArray(body.facts),
        narrative: typeof body.narrative === "string" ? body.narrative : "",
        concepts: asStringArray(body.concepts),
        files: asStringArray(body.files),
        importance,
        ...(typeof body.confidence === "number" ? { confidence: body.confidence } : {}),
      };

      await kv.set(KV.observations(sessionId), id, obs);

      // Best-effort indexing — failures are logged but don't fail the
      // request. A bulk import can reindex afterward via rebuild-graph.sh
      // if any of these go silently wrong.
      try {
        getSearchIndex().add(obs);
      } catch (err) {
        logger.warn("backfill: BM25 add failed", {
          obsId: id,
          error: err instanceof Error ? err.message : String(err),
        });
      }
      await vectorIndexAddGuarded(
        id,
        sessionId,
        obs.title + " " + (obs.narrative || ""),
        { kind: "synthetic", logId: id },
      );

      // Bump the session's observationCount so the viewer's session
      // panel doesn't lie.
      const existing = await kv.get<Session>(KV.sessions, sessionId).catch(() => null);
      if (existing) {
        await kv.update(KV.sessions, sessionId, [
          { type: "set", path: "observationCount", value: (existing.observationCount || 0) + 1 },
          { type: "set", path: "updatedAt", value: new Date().toISOString() },
        ]);
      }
      return { status_code: 201, body: { observationId: id } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::backfill::observation",
    config: { api_path: "/agentmemory/backfill/observation", http_method: "POST" },
  });

  // ---------- POST /agentmemory/backfill/vectors-missing ----------
  // Targeted re-embed pass for observations/memories whose vector
  // representation is missing (e.g. live-API failures left them
  // un-embedded). Walks the whole KV, skips anything already in the
  // vector index, embeds the rest. Cost scales with the number of
  // missing entries, not the total corpus.
  sdk.registerFunction(
    "api::backfill::vectors-missing",
    async (req: ApiRequest): Promise<Response> => {
      if (!backfillEnabled()) return disabledResponse();
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const body = (req.body ?? {}) as Record<string, unknown>;
      const maxToEmbed =
        Number.isInteger(body.maxToEmbed) && (body.maxToEmbed as number) > 0
          ? (body.maxToEmbed as number)
          : undefined;
      // Refuse uncapped calls. Without a positive maxToEmbed an
      // unattended invocation will walk the full corpus and queue
      // every missing embedding, which can run for hours and
      // saturate the daemon's event loop. Smoke probes that pass
      // `{}` historically tripped this footgun.
      if (maxToEmbed === undefined) {
        return {
          status_code: 400,
          body: {
            error:
              "maxToEmbed is required (positive integer). To prevent runaway full-corpus backfills, this endpoint refuses uncapped calls.",
          },
        };
      }
      const result = await rebuildVectorMissing(kv, { maxToEmbed });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::backfill::vectors-missing",
    config: { api_path: "/agentmemory/backfill/vectors-missing", http_method: "POST" },
  });

  // ---------- POST /agentmemory/backfill/summary ----------
  sdk.registerFunction(
    "api::backfill::summary",
    async (req: ApiRequest): Promise<Response> => {
      if (!backfillEnabled()) return disabledResponse();
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;

      const body = (req.body ?? {}) as Record<string, unknown>;
      const sessionId = asString(body.sessionId);
      const project = asString(body.project);
      const title = asString(body.title);
      if (!sessionId || !project || !title) {
        return {
          status_code: 400,
          body: { error: "sessionId, project, and title are required strings" },
        };
      }
      const createdAt = asString(body.createdAt) || new Date().toISOString();
      const observationCount =
        typeof body.observationCount === "number" && Number.isFinite(body.observationCount)
          ? body.observationCount
          : 0;

      const summary: SessionSummary = {
        sessionId,
        project,
        createdAt,
        title: title.slice(0, 200),
        narrative: typeof body.narrative === "string" ? body.narrative : "",
        keyDecisions: asStringArray(body.keyDecisions),
        filesModified: asStringArray(body.filesModified),
        concepts: asStringArray(body.concepts),
        observationCount,
      };
      await kv.set(KV.summaries, sessionId, summary);
      return { status_code: 201, body: { sessionId } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::backfill::summary",
    config: { api_path: "/agentmemory/backfill/summary", http_method: "POST" },
  });
}
