import type { StateKV } from "./kv.js";
import { KV } from "./schema.js";
import { VectorIndex } from "./vector-index.js";
import { logger } from "../logger.js";

// Per-observation embedding storage. Source-of-truth for the vector
// index: each observation's embedding is persisted as its own KV record,
// sharded across `EMBEDDING_SHARD_COUNT` scopes (mem:emb:00 ..). The
// chunked vectors.meta+vectors.chunk.N pair stays as a fast-boot cache,
// but if it's missing or corrupt we can rebuild from these scopes without
// re-embedding (which would cost provider tokens or hours of local GPU).
// See [[project-2026-05-22-vector-persistence-followups]].
//
// Sharding rationale: iii-engine's file-based KV adapter rewrites the
// whole scope file on every state::set. A single mem:emb scope ballooned
// to ~1 GB after ~200k entries on this install, and per-write rewrites
// of that file took >180s under live load — every persistEmbedding
// timed out (see also [[project-2026-05-22-vector-index-hang]] for the
// related chunked-persistence fix). Splitting into 32 hash-bucketed
// scopes drops the per-write file size to ~30 MB, well within what
// iii-engine handles for normal observation scopes.

const EMBEDDING_SHARD_COUNT = 32;

export type StoredEmbedding = {
  obsId: string;
  sessionId: string;
  // Float32Array.buffer encoded as base64. Per-record, so each KV value
  // stays small (typically <50 KB) and can never trip V8's max single-
  // string length. Binary transport would be cleaner but requires
  // iii-sdk support; rejected — iii-sdk JSON-stringifies everything.
  embedding: string;
};

// Hash an obsId to a shard index in [0, EMBEDDING_SHARD_COUNT). agentmemory
// IDs are `<prefix>_<base36>_<hex12>` (generateId in src/state/schema.ts),
// so the trailing hex is uniformly distributed — taking 2 hex chars gives
// 0..255 which is then mod-reduced. Falls back to a chained-add hash for
// IDs that don't end in hex (defensive; not expected in practice).
function shardIndex(obsId: string): number {
  if (obsId.length >= 2) {
    const tail = obsId.slice(-2);
    const n = parseInt(tail, 16);
    if (Number.isFinite(n) && n >= 0) return n % EMBEDDING_SHARD_COUNT;
  }
  let h = 0;
  for (let i = 0; i < obsId.length; i++) {
    h = ((h << 5) - h + obsId.charCodeAt(i)) | 0;
  }
  return Math.abs(h) % EMBEDDING_SHARD_COUNT;
}

function shardScope(obsId: string): string {
  const idx = shardIndex(obsId);
  return `${KV.embeddings}:${idx.toString().padStart(2, "0")}`;
}

function allShardScopes(): string[] {
  const scopes: string[] = [];
  for (let i = 0; i < EMBEDDING_SHARD_COUNT; i++) {
    scopes.push(`${KV.embeddings}:${i.toString().padStart(2, "0")}`);
  }
  return scopes;
}

// Cheap "is the per-obs store populated at all" probe — short-circuits on
// the first non-empty shard. Used by IndexPersistence to gate the
// one-time backfill migration without paying a full kv.list per shard.
export async function isEmbeddingStoreEmpty(kv: StateKV): Promise<boolean> {
  for (const scope of allShardScopes()) {
    try {
      const records = await kv.list<unknown>(scope);
      if (Array.isArray(records) && records.length > 0) return false;
    } catch {
      // Treat this shard as empty for the check and continue; if every
      // shard either errors or is empty the caller still sees "empty"
      // and we proceed with the backfill — safe-ish default since the
      // backfill itself is idempotent (writes by obsId).
    }
  }
  return true;
}

function float32ToBase64(arr: Float32Array): string {
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString(
    "base64",
  );
}

function base64ToFloat32(b64: string): Float32Array | null {
  try {
    const buf = Buffer.from(b64, "base64");
    if (buf.byteLength % 4 !== 0) return null;
    // Copy out of the pooled Buffer so the Float32Array owns its memory
    // — otherwise the underlying ArrayBuffer may be reused by Node's
    // Buffer pool and corrupt the stored vectors.
    const copy = new ArrayBuffer(buf.byteLength);
    Buffer.from(copy).set(buf);
    return new Float32Array(copy);
  } catch {
    return null;
  }
}

export async function persistEmbedding(
  kv: StateKV,
  obsId: string,
  sessionId: string,
  embedding: Float32Array,
): Promise<void> {
  const record: StoredEmbedding = {
    obsId,
    sessionId,
    embedding: float32ToBase64(embedding),
  };
  await kv.set(shardScope(obsId), obsId, record);
}

export async function removeEmbedding(
  kv: StateKV,
  obsId: string,
): Promise<void> {
  await kv.delete(shardScope(obsId), obsId);
}

// Walk every shard and collect records into a single array. Failures on
// individual shards are logged and skipped so a single corrupted shard
// can't take down rebuild.
async function listAllShardedRecords(
  kv: StateKV,
  context: string,
): Promise<StoredEmbedding[]> {
  const all: StoredEmbedding[] = [];
  let shardFailures = 0;
  for (const scope of allShardScopes()) {
    try {
      const records = await kv.list<StoredEmbedding>(scope);
      if (Array.isArray(records) && records.length > 0) {
        for (const r of records) all.push(r);
      }
    } catch (err) {
      shardFailures++;
      if (shardFailures <= 3) {
        logger.warn(`embedding-store: shard list failed (${context})`, {
          scope,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }
  }
  return all;
}

// Load the entire embedding store into a fresh VectorIndex. Returns null
// if the store is empty or all shards failed — caller falls back to the
// in-memory index (possibly empty) or to the expensive re-embed path.
export async function loadVectorIndexFromStore(
  kv: StateKV,
): Promise<VectorIndex | null> {
  const records = await listAllShardedRecords(kv, "load");
  if (records.length === 0) return null;

  const idx = new VectorIndex();
  let skipped = 0;
  for (const r of records) {
    if (
      !r ||
      typeof r.obsId !== "string" ||
      typeof r.sessionId !== "string" ||
      typeof r.embedding !== "string"
    ) {
      skipped++;
      continue;
    }
    const emb = base64ToFloat32(r.embedding);
    if (!emb) {
      skipped++;
      continue;
    }
    idx.add(r.obsId, r.sessionId, emb);
  }
  if (skipped > 0) {
    logger.warn("embedding-store: skipped malformed records on load", {
      skipped,
      loaded: idx.size,
    });
  }
  return idx.size > 0 ? idx : null;
}

// Lightweight variant of loadVectorIndexFromStore for the hybrid rebuild
// path. Returns just the embeddings keyed by obsId — sessionId is
// re-read from the live observation walk in rebuildIndex(), so a session
// re-key between the original write and the rebuild doesn't strand the
// vector under a stale sessionId. Returns null if all shards are empty.
export async function loadEmbeddingMap(
  kv: StateKV,
): Promise<Map<string, Float32Array> | null> {
  const records = await listAllShardedRecords(kv, "loadMap");
  if (records.length === 0) return null;

  const map = new Map<string, Float32Array>();
  let skipped = 0;
  for (const r of records) {
    if (
      !r ||
      typeof r.obsId !== "string" ||
      typeof r.embedding !== "string"
    ) {
      skipped++;
      continue;
    }
    const emb = base64ToFloat32(r.embedding);
    if (!emb) {
      skipped++;
      continue;
    }
    map.set(r.obsId, emb);
  }
  if (skipped > 0) {
    logger.warn("embedding-store: loadEmbeddingMap skipped malformed records", {
      skipped,
      loaded: map.size,
    });
  }
  return map.size > 0 ? map : null;
}

// First-boot migration: chunked load succeeded but the per-obs store is
// empty (legacy install upgrading to per-obs persistence). Backfill the
// store from the in-memory vector index so future rebuilds can use it.
// Yields between writes so the event loop keeps draining the REST accept
// queue — same lesson as the chunked save in IndexPersistence.
export async function backfillEmbeddingStoreFromIndex(
  kv: StateKV,
  vector: VectorIndex,
): Promise<{ written: number; failed: number }> {
  const internal = vector as unknown as {
    vectors: Map<string, { embedding: Float32Array; sessionId: string }>;
  };
  let written = 0;
  let failed = 0;
  for (const [obsId, entry] of internal.vectors) {
    try {
      await persistEmbedding(kv, obsId, entry.sessionId, entry.embedding);
      written++;
    } catch (err) {
      failed++;
      if (failed <= 3) {
        logger.warn("embedding-store: backfill write failed", {
          obsId,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
  return { written, failed };
}
