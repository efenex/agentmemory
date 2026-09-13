import type { ISdk } from 'iii-sdk'
import type { CompactSearchResult, CompressedObservation, Memory, SearchResult, Session } from '../types.js'
import { KV } from '../state/schema.js'
import { StateKV } from '../state/kv.js'
import { SearchIndex } from '../state/search-index.js'
import { VectorIndex } from '../state/vector-index.js'
import { loadEmbeddingMap, persistEmbedding } from '../state/embedding-store.js'
import type { EmbeddingProvider } from '../types.js'
import { memoryToObservation } from '../state/memory-utils.js'
import { recordAccessBatch } from './access-tracker.js'
import { logger } from "../logger.js";
import { getAgentId, isAgentScopeIsolated } from "../config.js";

let index: SearchIndex | null = null
let vectorIndex: VectorIndex | null = null
let currentEmbeddingProvider: EmbeddingProvider | null = null
// KV handle for per-obs embedding persistence. Set during boot in
// src/index.ts alongside setVectorIndex; nullable so test harnesses and
// the BM25-only mode don't need a stub.
let stateKvForEmbeddings: StateKV | null = null

// Hybrid ranking hook for mem::search. Wired by index.ts once the
// hybrid searcher exists (it is constructed after this module's
// registration runs). When set and the vector index has entries,
// mem::search ranks candidates through the full BM25+vector+graph
// fusion instead of BM25 alone — previously only mem::smart-search got
// hybrid ranking while the primary recall surface stayed keyword-only.
type HybridRanker = (
  query: string,
  limit: number,
) => Promise<Array<{ observation: CompressedObservation; sessionId: string; combinedScore: number }>>
let hybridRanker: HybridRanker | null = null

export function setHybridRanker(fn: HybridRanker | null): void {
  hybridRanker = fn
}

// Dedupes the lazy cold-start rebuild kicked off from the mem::search
// request path. A full rebuildIndex walks every observation across every
// session, so N concurrent queries against an empty index would each
// launch their own rebuild and saturate the engine invocation pool. The
// first query with an empty index starts one rebuild and shares its
// promise; concurrent queries await the same rebuild instead of spawning
// duplicates. The boot-time rebuild in index.ts is unaffected.
let rebuildPromise: Promise<number> | null = null

let memoryIndexReady = false
export function isMemoryIndexReady(): boolean {
  return memoryIndexReady
}

export function getSearchIndex(): SearchIndex {
  if (!index) index = new SearchIndex()
  return index
}

export function setVectorIndex(idx: VectorIndex | null): void {
  vectorIndex = idx
}

export function getVectorIndex(): VectorIndex | null {
  return vectorIndex
}

export function setStateKvForEmbeddings(kv: StateKV | null): void {
  stateKvForEmbeddings = kv
}

export function setEmbeddingProvider(provider: EmbeddingProvider | null): void {
  currentEmbeddingProvider = provider
}

export function getEmbeddingProvider(): EmbeddingProvider | null {
  return currentEmbeddingProvider
}

// Fire-and-forget per-obs embedding persistence. Logs+swallows so a
// transient state::set failure can't break the upstream observe/compress
// path; on next boot the chunked-load fast path still works, and the
// per-obs store will be backfilled from in-memory the next time
// IndexPersistence.load() runs (see embedding-store.ts).
function persistEmbeddingBg(
  obsId: string,
  sessionId: string,
  embedding: Float32Array,
): void {
  const kv = stateKvForEmbeddings
  if (!kv) return
  persistEmbedding(kv, obsId, sessionId, embedding).catch((err) => {
    logger.warn("embedding-store: persistEmbedding failed", {
      obsId,
      error: err instanceof Error ? err.message : String(err),
    })
  })
}

export function vectorIndexRemove(id: string): void {
  vectorIndex?.remove(id);
}

// Persistence sync hook. Without this, index removals only live in
// memory; a crash/SIGKILL before graceful shutdown reloads a stale
// snapshot at boot and the deleted entry resurrects in the index.
// Wired by src/index.ts after IndexPersistence is constructed; no-op
// until then so unit tests that exercise the delete paths in
// isolation don't need to wire persistence.
let indexPersistence: {
  scheduleSave: () => void;
  save: () => Promise<void>;
} | null = null;

export function setIndexPersistence(
  p: { scheduleSave: () => void; save: () => Promise<void> } | null,
): void {
  indexPersistence = p;
}

export function scheduleIndexSave(): void {
  indexPersistence?.scheduleSave();
}

// Synchronous flush variant for delete paths. The debounced
// scheduleSave is fine for adds (chatty), but a hard process exit
// inside the 5s debounce window would lose deletes and resurrect
// removed entries on next boot. Deletes are infrequent enough that
// awaiting a single write per operation is acceptable. save() catches
// its own errors via IndexPersistence.logFailure, so this resolves
// even when persistence fails — callers must not treat a failed
// flush as a fatal error on the delete itself (the KV delete already
// committed before this is invoked).
export async function flushIndexSave(): Promise<void> {
  await indexPersistence?.save();
}

// Hard cap on embedding input length. Most providers cap input around
// 8k tokens (~32k chars at ~4 chars/token). Truncate defensively so a
// huge memory.content can't 400 the embed call or blow context budget
// on a single doc. 16k chars ≈ 4k tokens, safely under every provider.
const EMBED_MAX_CHARS = 16_000

export function clipEmbedInput(text: string): string {
  if (text.length <= EMBED_MAX_CHARS) return text
  return text.slice(0, EMBED_MAX_CHARS)
}

// Single guarded vector-index write. Returns true on success. Logs and
// no-ops on:
//   - dimension mismatch (mis-configured provider would silently corrupt
//     the index per #248 otherwise — guarded at persistence load there;
//     this is the symmetric guard at the write site)
//   - embed throwing (network, rate limit, provider down)
// Always soft-fails so a downed embedder doesn't break the upstream save.
export async function vectorIndexAddGuarded(
  id: string,
  sessionId: string,
  text: string,
  context: { kind: "memory" | "observation" | "synthetic"; logId: string },
): Promise<boolean> {
  const vi = vectorIndex
  const ep = currentEmbeddingProvider
  if (!vi || !ep) return false
  try {
    const embedding = await ep.embed(clipEmbedInput(text))
    if (embedding.length !== ep.dimensions) {
      logger.warn("vector-index add: dimension mismatch — skipping", {
        kind: context.kind,
        id: context.logId,
        provider: ep.name,
        expected: ep.dimensions,
        received: embedding.length,
      })
      return false
    }
    vi.add(id, sessionId, embedding)
    persistEmbeddingBg(id, sessionId, embedding)
    return true
  } catch (err) {
    logger.warn("vector-index add: embed failed — skipping", {
      kind: context.kind,
      id: context.logId,
      provider: ep.name,
      error: err instanceof Error ? err.message : String(err),
    })
    return false
  }
}

// Batched variant: calls EmbeddingProvider.embedBatch ONCE for the whole
// batch, then writes each resulting vector. Use this for bulk paths
// (rebuildIndex, future bulk-add APIs) where per-item serial awaits
// dominate wallclock. A batch of N has roughly the latency of a single
// embed (network + GPU setup amortized), so backfilling a 500k-obs
// corpus drops from days to hours on a per-batch endpoint like vLLM.
//
// Per-item failure shape:
//   - whole-batch network/provider error → all skipped, single warn line
//   - per-item dimension mismatch → that item skipped, others continue
export async function vectorIndexAddBatchGuarded(
  items: Array<{
    id: string
    sessionId: string
    text: string
    context: { kind: "memory" | "observation" | "synthetic"; logId: string }
  }>,
): Promise<{ ok: number; fail: number }> {
  const vi = vectorIndex
  const ep = currentEmbeddingProvider
  if (!vi || !ep || items.length === 0) return { ok: 0, fail: 0 }

  let embeddings: Float32Array[]
  try {
    embeddings = await ep.embedBatch(items.map((i) => clipEmbedInput(i.text)))
  } catch (err) {
    logger.warn("vector-index add batch: embed failed — skipping batch", {
      batchSize: items.length,
      provider: ep.name,
      error: err instanceof Error ? err.message : String(err),
    })
    return { ok: 0, fail: items.length }
  }

  if (embeddings.length !== items.length) {
    logger.warn(
      "vector-index add batch: provider returned wrong length — skipping batch",
      {
        batchSize: items.length,
        returned: embeddings.length,
        provider: ep.name,
      },
    )
    return { ok: 0, fail: items.length }
  }

  let ok = 0
  let fail = 0
  for (let i = 0; i < items.length; i++) {
    const item = items[i]
    const embedding = embeddings[i]
    if (embedding.length !== ep.dimensions) {
      logger.warn("vector-index add batch: dimension mismatch — skipping item", {
        kind: item.context.kind,
        id: item.context.logId,
        provider: ep.name,
        expected: ep.dimensions,
        received: embedding.length,
      })
      fail++
      continue
    }
    try {
      vi.add(item.id, item.sessionId, embedding)
      persistEmbeddingBg(item.id, item.sessionId, embedding)
      ok++
    } catch (err) {
      logger.warn("vector-index add batch: index write failed — skipping item", {
        kind: item.context.kind,
        id: item.context.logId,
        error: err instanceof Error ? err.message : String(err),
      })
      fail++
    }
  }
  return { ok, fail }
}

// Embed-batch size for rebuild. Each item is one /v1/embeddings call's
// `input` array element; the provider sees the whole batch as one HTTP
// round-trip. 32 fits comfortably under typical per-request token budgets
// (32 × ~110 tok/item ≈ 3.5k tokens) and gets close to per-call
// throughput for GPU-backed endpoints (vLLM, Triton, etc.). Override via
// REBUILD_EMBED_BATCH_SIZE for endpoints that prefer smaller/larger
// batches. Set to 1 to fall back to the legacy per-item path.
const DEFAULT_REBUILD_EMBED_BATCH = 32

function getRebuildEmbedBatchSize(): number {
  const raw = process.env.REBUILD_EMBED_BATCH_SIZE
  if (!raw) return DEFAULT_REBUILD_EMBED_BATCH
  const n = parseInt(raw, 10)
  return Number.isFinite(n) && n > 0 ? n : DEFAULT_REBUILD_EMBED_BATCH
}

// How many embed batches the rebuild can have in flight at once.
// Default 1 (sequential, current behavior). Bump via
// REBUILD_EMBED_CONCURRENCY for vLLM/Triton-backed endpoints that can
// service multiple concurrent /v1/embeddings POSTs in parallel (e.g.
// set vLLM `--max-num-seqs 8` and pair with concurrency=4 here).
// Still consumed by rebuildVectorMissing() below.
const DEFAULT_REBUILD_EMBED_CONCURRENCY = 1

function getRebuildEmbedConcurrency(): number {
  const raw = process.env.REBUILD_EMBED_CONCURRENCY
  if (!raw) return DEFAULT_REBUILD_EMBED_CONCURRENCY
  const n = parseInt(raw, 10)
  return Number.isFinite(n) && n > 0 ? n : DEFAULT_REBUILD_EMBED_CONCURRENCY
}

// Reuse-store for the stored-embedding fast path, populated by
// rebuildIndex() before it walks anything and consumed by indexRecords()
// via enqueue(). It is module-level rather than a parameter because
// upstream's indexRecords() is the shared entry point for the importers
// too, and those have no StateKV to load a map from — they simply leave
// this null and embed normally.
//
// The win it buys: on a populated install every observation already has a
// persisted vector under KV.embeddings, so a rebuild becomes "one kv.list
// plus in-memory copies" instead of hours of provider re-embeds. Entries
// whose dimension does not match the ACTIVE provider (i.e. the provider
// was swapped since the vector was written) deliberately miss this path
// and fall through to a real embed, so the rebuilt index never mixes two
// geometries.
let rebuildEmbeddingCache: Map<string, number[]> | null = null
let rebuildExpectedDim = 0
let rebuildFromStore = 0
let rebuildReembedQueued = 0

// Shared BM25 + batched-vector indexing for a set of records. The full
// rebuild and every import path (export-import, jsonl replay) funnel
// through this so they index identically and none can silently skip the
// vector side. It does NOT clear the index — callers that rebuild clear
// first; importers add. When no embedding provider is configured it skips
// the vector enqueue entirely, so a keyless install never allocates embed
// jobs it would immediately discard.
export async function indexRecords(
  observations: CompressedObservation[],
  memories: Memory[],
): Promise<number> {
  const idx = getSearchIndex()
  const vectorEnabled = Boolean(vectorIndex && currentEmbeddingProvider)
  const batchSize = getRebuildEmbedBatchSize()
  type EmbedJob = {
    id: string
    sessionId: string
    text: string
    context: { kind: "memory" | "observation" | "synthetic"; logId: string }
  }
  const pending: EmbedJob[] = []
  const flush = async (): Promise<void> => {
    if (pending.length === 0) return
    await vectorIndexAddBatchGuarded(pending)
    pending.length = 0
  }
  const enqueue = async (job: EmbedJob): Promise<void> => {
    if (!vectorEnabled) return
    // Fast path: vector for this obsId is already persisted under
    // KV.embeddings — copy it into the in-memory index and skip the
    // provider call entirely. Only ever populated on the rebuild path
    // (see rebuildEmbeddingCache); importers leave it null.
    if (rebuildEmbeddingCache && vectorIndex) {
      const stored = rebuildEmbeddingCache.get(job.id)
      if (stored && (rebuildExpectedDim === 0 || stored.length === rebuildExpectedDim)) {
        vectorIndex.add(job.id, job.sessionId, stored)
        rebuildFromStore++
        return
      }
    }
    pending.push(job)
    rebuildReembedQueued++
    if (pending.length >= batchSize) await flush()
  }

  let count = 0
  for (const memory of memories) {
    if (memory.isLatest === false) continue
    if (!memory.title || !memory.content) continue
    idx.add(memoryToObservation(memory))
    await enqueue({
      id: memory.id,
      sessionId: memory.sessionIds?.[0] ?? 'memory',
      text: memory.title + ' ' + memory.content,
      context: { kind: "memory", logId: memory.id },
    })
    count++
  }
  for (const obs of observations) {
    if (!obs.title || !obs.narrative) continue
    idx.add(obs)
    await enqueue({
      id: obs.id,
      sessionId: obs.sessionId,
      text: obs.title + ' ' + obs.narrative,
      context: { kind: "observation", logId: obs.id },
    })
    count++
  }
  await flush()
  return count
}

export async function rebuildIndex(kv: StateKV): Promise<number> {
  const idx = getSearchIndex()
  idx.clear()
  memoryIndexReady = false

  // BM25 clear above wipes stale doc entries; the vector index has the
  // symmetric concern — memories/observations deleted between runs
  // would leave orphan embeddings here forever. Clear both before the
  // repopulation loops run, so BM25 and vector stay in sync.
  vectorIndex?.clear()

  // Populate the stored-embedding reuse cache for the duration of this
  // rebuild. indexRecords() consults it via enqueue(); it is cleared in
  // the finally below so importers (which share indexRecords) never see a
  // stale map from a previous rebuild.
  const tEmbMap = Date.now()
  rebuildFromStore = 0
  rebuildReembedQueued = 0
  rebuildExpectedDim = currentEmbeddingProvider?.dimensions ?? 0
  rebuildEmbeddingCache = vectorIndex
    ? await loadEmbeddingMap(kv).catch((err) => {
        logger.warn("rebuildIndex: loadEmbeddingMap failed; falling back to re-embed", {
          error: err instanceof Error ? err.message : String(err),
        })
        return null
      })
    : null
  logger.info("rebuildIndex: loadEmbeddingMap done", {
    durationMs: Date.now() - tEmbMap,
    size: rebuildEmbeddingCache?.size ?? 0,
  })

  try {

  // Memories live in their own KV scope outside per-session observation
  // scopes, so they need a separate walk. Without this, mem::remember
  // entries vanish from BM25 on every restart even after the live-write
  // fix in remember.ts.
  let memories: Memory[] = []
  let memoriesLoaded = false
  try {
    memories = await kv.list<Memory>(KV.memories)
    memoriesLoaded = true
  } catch (err) {
    logger.warn('rebuildIndex: failed to load memories', {
      error: err instanceof Error ? err.message : String(err),
    })
  }

  const tListStart = Date.now()
  const sessions = await kv.list<Session>(KV.sessions)
  logger.info("rebuildIndex: kv.list(sessions) returned", {
    durationMs: Date.now() - tListStart,
    length: Array.isArray(sessions) ? sessions.length : null,
  })
  const failedSessions: string[] = []
  // Index each session chunk as it loads instead of accumulating every
  // observation first, so peak memory stays bounded to one chunk.
  let indexed = 0
  for (let batch = 0; batch < sessions.length; batch += 10) {
    const chunk = sessions.slice(batch, batch + 10)
    const results = await Promise.all(
      chunk.map(async (s) => {
        try {
          return await kv.list<CompressedObservation>(KV.observations(s.id))
        } catch {
          failedSessions.push(s.id)
          return [] as CompressedObservation[]
        }
      })
    )
    const chunkObs = results.flat()
    if (chunkObs.length > 0) {
      indexed += await indexRecords(chunkObs, [])
    }
  }
  if (failedSessions.length > 0) {
    logger.warn('rebuildIndex: failed to load observations for sessions', { failedSessions })
  }

  indexed += await indexRecords([], memories)
  if (memoriesLoaded) memoryIndexReady = true

  // How much of the rebuild came from persisted vectors vs a real
  // provider call. This is the number to read when a rebuild is
  // unexpectedly slow or expensive: a low fromStore means the embedding
  // geometry changed (provider swap) and the whole corpus is re-embedding.
  if (vectorIndex) {
    logger.info("rebuildIndex: vector source breakdown", {
      total: indexed,
      fromStore: rebuildFromStore,
      reembedQueued: rebuildReembedQueued,
      storeAvailable: rebuildEmbeddingCache !== null,
      storeSize: rebuildEmbeddingCache?.size ?? 0,
    })
  }
  return indexed
  } finally {
    // Always drop the map: it can hold one float array per observation
    // (hundreds of MB at this corpus size) and indexRecords is shared
    // with the import paths, which must not inherit it.
    rebuildEmbeddingCache = null
  }
}

// Targeted re-embed: walks every memory + observation and only embeds
// rows whose obsId is NOT already in the vector index. Used to recover
// from a transient embedding-provider failure that left a chunk of the
// corpus un-embedded (e.g. dimension-mismatch bug that silently
// dropped vectors for live observations between two migrations).
//
// Unlike rebuildIndex(), this:
//   - does NOT clear the existing vector / BM25 indexes
//   - skips any obsId whose vector is already present
//   - returns a summary of what it found vs what it embedded
//
// Cost scales with the number of missing entries, not the total corpus.
export async function rebuildVectorMissing(
  kv: StateKV,
  opts: { maxToEmbed?: number } = {},
): Promise<{
  checked: number;
  attempted: number;
  embedded: number;
  failed: number;
  skipped: number;
  capped: boolean;
}> {
  if (!vectorIndex) {
    return {
      checked: 0,
      attempted: 0,
      embedded: 0,
      failed: 0,
      skipped: 0,
      capped: false,
    };
  }
  const maxToEmbed =
    Number.isInteger(opts.maxToEmbed) && (opts.maxToEmbed as number) > 0
      ? (opts.maxToEmbed as number)
      : Number.MAX_SAFE_INTEGER;
  const batchSize = getRebuildEmbedBatchSize();
  const concurrency = getRebuildEmbedConcurrency();
  type EmbedJob = {
    id: string;
    sessionId: string;
    text: string;
    context: { kind: "memory" | "observation" | "synthetic"; logId: string };
  };
  let pending: EmbedJob[] = [];
  let checked = 0;
  let attempted = 0;
  let embedded = 0;
  let failed = 0;
  let skipped = 0;
  // In-flight batch promises. We cap the set at `concurrency`; when
  // full, `enqueueIfMissing` awaits `Promise.race` to free a slot
  // before pushing the next batch — keeps memory bounded while letting
  // the embedding endpoint service multiple POSTs simultaneously.
  const inFlight = new Set<Promise<void>>();
  let lastProgressAttempted = 0;

  const dispatch = (batch: EmbedJob[]): void => {
    const p = vectorIndexAddBatchGuarded(batch).then(
      (res) => {
        inFlight.delete(p);
        embedded += res.ok;
        failed += res.fail;
        // Throttled progress: log roughly every 10 batches' worth so
        // we don't drown stderr at high concurrency.
        if (attempted - lastProgressAttempted >= batchSize * 10) {
          lastProgressAttempted = attempted;
          logger.info("rebuildVectorMissing: progress", {
            checked,
            attempted,
            embedded,
            failed,
            skipped,
            inFlight: inFlight.size,
          });
        }
      },
      (err) => {
        inFlight.delete(p);
        failed += batch.length;
        logger.warn("rebuildVectorMissing: batch promise rejected", {
          batchSize: batch.length,
          error: err instanceof Error ? err.message : String(err),
        });
      },
    );
    inFlight.add(p);
  };

  const flush = async (): Promise<void> => {
    if (pending.length === 0) return;
    const batch = pending;
    pending = [];
    dispatch(batch);
    if (inFlight.size >= concurrency) {
      await Promise.race(inFlight);
    }
  };
  let capped = false;
  const enqueueIfMissing = async (job: EmbedJob): Promise<void> => {
    checked++;
    if (vectorIndex!.has(job.id)) {
      skipped++;
      return;
    }
    if (attempted >= maxToEmbed) {
      capped = true;
      return;
    }
    attempted++;
    pending.push(job);
    if (pending.length >= batchSize) await flush();
  };

  try {
    const memories = await kv.list<Memory>(KV.memories);
    for (const memory of memories) {
      if (capped) break;
      if (memory.isLatest === false) continue;
      if (!memory.title || !memory.content) continue;
      await enqueueIfMissing({
        id: memory.id,
        sessionId: memory.sessionIds[0] ?? "memory",
        text: memory.title + " " + memory.content,
        context: { kind: "memory", logId: memory.id },
      });
    }
  } catch (err) {
    logger.warn("rebuildVectorMissing: failed to load memories", {
      error: err instanceof Error ? err.message : String(err),
    });
  }

  const sessions = await kv.list<Session>(KV.sessions);
  outer: for (let batch = 0; batch < sessions.length; batch += 10) {
    if (capped) break;
    const chunk = sessions.slice(batch, batch + 10);
    const obsLists = await Promise.all(
      chunk.map(async (s) => {
        try {
          return await kv.list<CompressedObservation>(KV.observations(s.id));
        } catch {
          return [] as CompressedObservation[];
        }
      }),
    );
    for (const observations of obsLists) {
      for (const obs of observations) {
        if (capped) break outer;
        if (!obs.title || !obs.narrative) continue;
        await enqueueIfMissing({
          id: obs.id,
          sessionId: obs.sessionId,
          text: obs.title + " " + obs.narrative,
          context: { kind: "observation", logId: obs.id },
        });
      }
    }
  }

  await flush();
  // Drain any remaining in-flight batches before reporting "done".
  if (inFlight.size > 0) {
    await Promise.allSettled(inFlight);
  }
  logger.info("rebuildVectorMissing: done", {
    checked,
    attempted,
    embedded,
    failed,
    skipped,
    capped,
    concurrency,
  });
  return { checked, attempted, embedded, failed, skipped, capped };
}

export function registerSearchFunction(sdk: ISdk, kv: StateKV): void {
  sdk.registerFunction(
    "mem::vector::backfill-missing",
    async (data: { maxToEmbed?: number } = {}): Promise<{
      success: true;
      checked: number;
      attempted: number;
      embedded: number;
      failed: number;
      skipped: number;
      capped: boolean;
    }> => {
      const result = await rebuildVectorMissing(kv, {
        maxToEmbed: data.maxToEmbed,
      });
      return { success: true, ...result };
    },
  );

  sdk.registerFunction(
    'mem::search',
    async (data: {
      query: string
      limit?: number
      project?: string
      cwd?: string
      format?: string
      token_budget?: number
      agentId?: string
    }) => {
      const idx = getSearchIndex()

      // Input validation / normalization.
      if (typeof data?.query !== 'string' || !data.query.trim()) {
        throw new Error('mem::search: query must be a non-empty string')
      }
      const query = data.query.trim()
      const MAX_LIMIT = 100
      let effectiveLimit = 20
      if (data.limit !== undefined) {
        if (!Number.isInteger(data.limit) || data.limit < 1) {
          throw new Error('mem::search: limit must be a positive integer')
        }
        effectiveLimit = Math.min(data.limit, MAX_LIMIT)
      }
      const projectFilter = typeof data.project === 'string' && data.project.trim().length > 0 ? data.project.trim() : undefined
      const cwdFilter = typeof data.cwd === 'string' && data.cwd.trim().length > 0 ? data.cwd.trim() : undefined
      // #817: agent-scope isolation. mem::search backs REST /search,
      // memory_recall and recall_context. Without filtering here a
      // worker booted with AGENT_ID=B + AGENTMEMORY_AGENT_SCOPE=isolated
      // could read A's memories — the cross-agent leak the issue
      // documented. Mirrors the smart-search pattern: wildcard "*"
      // bypasses, explicit agentId pins, isolated mode falls back to
      // the worker's own AGENT_ID.
      //
      // Fail-closed: if isolated mode is on AND no explicit agentId
      // is given AND env AGENT_ID is unset, refuse the call rather
      // than silently dropping the filter. Allowing the call through
      // with filterAgentId=undefined is the same leak this fix is
      // supposed to close.
      const isolated = isAgentScopeIsolated();
      const explicitAgentId =
        typeof data.agentId === "string" && data.agentId.trim().length > 0
          ? data.agentId.trim()
          : undefined;
      const wildcardAgent = explicitAgentId === "*";
      const envAgentId = isolated ? getAgentId() : undefined;
      const filterAgentId = wildcardAgent
        ? undefined
        : explicitAgentId ?? envAgentId;
      if (
        isolated &&
        !wildcardAgent &&
        !explicitAgentId &&
        !envAgentId
      ) {
        throw new Error(
          "mem::search: AGENTMEMORY_AGENT_SCOPE=isolated is set but no " +
            "agent id is available (env AGENT_ID unset and no explicit " +
            "agentId in the call). Refusing to read cross-agent rows. " +
            'Pass agentId: "*" to opt in to a wildcard read.',
        );
      }
      const format = typeof data.format === 'string' ? data.format : 'full'
      if (!['full', 'compact', 'narrative'].includes(format)) {
        throw new Error("mem::search: format must be one of 'full', 'compact', or 'narrative'")
      }
      let tokenBudget: number | undefined
      if (data.token_budget !== undefined) {
        if (!Number.isInteger(data.token_budget) || data.token_budget < 1) {
          throw new Error('mem::search: token_budget must be a positive integer')
        }
        tokenBudget = data.token_budget
      }

      if (idx.size === 0) {
        // Share one rebuild across concurrent cold-start queries so they
        // don't each walk the whole corpus and saturate the pool.
        if (!rebuildPromise) {
          rebuildPromise = rebuildIndex(kv)
            .then((count) => {
              logger.info('Search index rebuilt', { entries: count })
              return count
            })
            .catch((err) => {
              logger.warn('Index rebuild failed', {
                error: err instanceof Error ? err.message : String(err),
              })
              return 0
            })
            .finally(() => {
              rebuildPromise = null
            })
        }
        await rebuildPromise
      }

      // When filtering by project/cwd, over-fetch from the index so the
      // post-filter still has a chance of returning `effectiveLimit` results.
      // Over-fetch whenever ANY post-index filter is active. agentId
      // is dropped after the observation/memory is loaded (BM25 index
      // doesn't carry it), so without the over-fetch isolated-mode
      // queries return underfilled pages when same-agent matches
      // rank lower than cross-agent ones in the hybrid score.
      const filtering = !!(projectFilter || cwdFilter || filterAgentId)
      const fetchLimit = filtering ? Math.max(effectiveLimit * 10, 100) : effectiveLimit
      // Hybrid results carry the observation the ranker already loaded,
      // so the load pass below doesn't refetch every record it just
      // enriched.
      let results: Array<{
        obsId: string
        sessionId: string
        score: number
        observation?: CompressedObservation
      }>
      if (hybridRanker && vectorIndex && vectorIndex.size > 0) {
        try {
          const hybrid = await hybridRanker(query, fetchLimit)
          results = hybrid.map((r) => ({
            obsId: r.observation.id,
            sessionId: r.sessionId,
            score: r.combinedScore,
            observation: r.observation,
          }))
        } catch (err) {
          logger.warn("hybrid ranking failed, falling back to keyword search", {
            error: err instanceof Error ? err.message : String(err),
          })
          results = idx.search(query, fetchLimit)
        }
      } else {
        results = idx.search(query, fetchLimit)
      }

      // Resolve session -> project/cwd once per sessionId we touch.
      const sessionCache = new Map<string, Session | null>()
      const loadSession = async (sessionId: string): Promise<Session | null> => {
        if (sessionCache.has(sessionId)) return sessionCache.get(sessionId)!
        const s = await kv.get<Session>(KV.sessions, sessionId)
        sessionCache.set(sessionId, s ?? null)
        return s ?? null
      }

      // Cache for memory project lookups. Memories indexed via mem::remember
      // use a synthetic sessionId ('memory' or the first real sessionId) that
      // either has no KV.sessions entry or belongs to a different project.
      // When loadSession returns null we fall through to a KV.memories probe
      // so project-filtered search can include or exclude them correctly.
      const memoryProjectCache = new Map<string, string | null>()
      const loadMemoryProject = async (obsId: string): Promise<string | null> => {
        if (memoryProjectCache.has(obsId)) return memoryProjectCache.get(obsId)!
        const mem = await kv.get<Memory>(KV.memories, obsId).catch(() => null)
        const proj = mem?.project ?? null
        memoryProjectCache.set(obsId, proj)
        return proj
      }

      // First pass: filter by session (sequential — benefits from session cache).
      // Memory entries with a synthetic sessionId take a secondary KV.memories
      // path so project filtering works correctly for them too.
      //
      // When agentId filtering is active we can't cap at effectiveLimit
      // here — the second pass (post-load) is what drops cross-agent
      // rows, and capping early would underfill the result page. Use
      // fetchLimit as the upper bound in that case; the final
      // truncation lives at the end of the second pass.
      const earlyCap = filterAgentId ? fetchLimit : effectiveLimit
      const candidates: typeof results = []
      for (const r of results) {
        if (candidates.length >= earlyCap) break
        if (filtering) {
          const s = await loadSession(r.sessionId)
          if (s) {
            if (projectFilter && s.project !== projectFilter) continue
            if (cwdFilter && s.cwd !== cwdFilter) continue
          } else {
            // Session not found. Two cases arrive here:
            //   1. Synthetic sessionId — memories indexed via mem::remember use
            //      sessionIds[0] ?? 'memory'. The string 'memory' has no session
            //      entry; neither does a real sessionId when sessionIds[0] happens
            //      to be a session from a different lifecycle. Probe KV.memories
            //      directly to get the memory's own project field.
            //   2. Deleted session — the session existed when the entry was indexed
            //      but was since evicted. The KV.memories probe returns null for
            //      these (they are observations, not memories), so memProject is
            //      null and the entry passes through as unscoped. This is the safe
            //      fallback: we lose the ability to filter but never incorrectly
            //      block a result whose session we can no longer verify.
            // In both cases, a null memProject means "project unknown — treat as
            // unscoped and let it through" to preserve backward-compatibility.
            if (projectFilter) {
              const memProject = await loadMemoryProject(r.obsId)
              if (memProject !== null && memProject !== projectFilter) continue
            }
            // cwd filter does not apply to unbound entries.
          }
        }
        candidates.push(r)
      }

      // Second pass: load observations in parallel. Fall back to
      // KV.memories when the observation lookup misses — entries indexed
      // via mem::remember live in the memories scope under a synthetic
      // sessionId, so the observation key never exists (#265).
      const obsResults = await Promise.all(
        candidates.map(async (r) => {
          if (r.observation) return r.observation
          const obs = await kv
            .get<CompressedObservation>(KV.observations(r.sessionId), r.obsId)
            .catch(() => null)
          if (obs) return obs
          const mem = await kv
            .get<Memory>(KV.memories, r.obsId)
            .catch(() => null)
          return mem ? memoryToObservation(mem) : null
        })
      )
      const enriched: SearchResult[] = []
      for (let i = 0; i < candidates.length; i++) {
        const obs = obsResults[i]
        if (!obs) continue
        // #817: enforce agent-scope after the observation/memory is
        // loaded. The BM25 index doesn't carry agentId so the filter
        // happens post-lookup. Wildcard ("*") and no-isolation paths
        // resolved filterAgentId=undefined upstream and pass through.
        if (filterAgentId !== undefined && obs.agentId !== filterAgentId) continue
        if (enriched.length >= effectiveLimit) break
        enriched.push({
          observation: obs,
          score: candidates[i].score,
          sessionId: candidates[i].sessionId,
        })
      }

      void recordAccessBatch(
        kv,
        enriched.map((r) => r.observation.id),
      )

      const estimateTokens = (value: unknown): number =>
        Math.max(1, Math.ceil(JSON.stringify(value).length / 3))

      const applyTokenBudget = <T>(items: T[]): {
        items: T[]
        used: number
        truncated: boolean
      } => {
        if (!tokenBudget) return { items, used: items.reduce((sum, item) => sum + estimateTokens(item), 0), truncated: false }
        const selected: T[] = []
        let used = 0
        for (const item of items) {
          const itemTokens = estimateTokens(item)
          if (used + itemTokens > tokenBudget) {
            return { items: selected, used, truncated: selected.length < items.length }
          }
          selected.push(item)
          used += itemTokens
        }
        return { items: selected, used, truncated: false }
      }

      if (format === 'compact') {
        const compactResults: CompactSearchResult[] = enriched.map((r) => ({
          obsId: r.observation.id,
          sessionId: r.sessionId,
          title: r.observation.title,
          type: r.observation.type,
          score: r.score,
          timestamp: r.observation.timestamp,
        }))
        const packed = applyTokenBudget(compactResults)
        return {
          format,
          results: packed.items,
          tokens_used: packed.used,
          tokens_budget: tokenBudget,
          truncated: packed.truncated,
        }
      }

      if (format === 'narrative') {
        const narrativeResults = enriched.map((r) => ({
          obsId: r.observation.id,
          sessionId: r.sessionId,
          title: r.observation.title,
          narrative: r.observation.narrative,
          score: r.score,
          timestamp: r.observation.timestamp,
        }))
        const packed = applyTokenBudget(narrativeResults)
        const text = packed.items
          .map((r, index) => `${index + 1}. ${r.title}\n${r.narrative}`)
          .join('\n\n')
        return {
          format,
          results: packed.items,
          text,
          tokens_used: packed.used,
          tokens_budget: tokenBudget,
          truncated: packed.truncated,
        }
      }

      const packed = applyTokenBudget(enriched)

      // Avoid logging raw cwd/project (host paths). Log only that filters were active.
      logger.info('Search completed', {
        query,
        results: packed.items.length,
        hasProjectFilter: !!projectFilter,
        hasCwdFilter: !!cwdFilter,
      })
      return {
        format,
        results: packed.items,
        tokens_used: packed.used,
        tokens_budget: tokenBudget,
        truncated: packed.truncated,
      }
    }
  )
}
