import { SearchIndex } from "./search-index.js";
import { VectorIndex } from "./vector-index.js";
import type { StateKV } from "./kv.js";
import { KV } from "./schema.js";
import { logger } from "../logger.js";
import {
  backfillEmbeddingStoreFromIndex,
  isEmbeddingStoreEmpty,
  loadVectorIndexFromStore,
} from "./embedding-store.js";

const DEBOUNCE_MS = 5000;
const FAILURE_LOG_THROTTLE_MS = 60_000;

// Chunked-vector persistence — see [[project-2026-05-22-vector-index-hang]].
// The old whole-index `state::set vectors` path stringified GBs of base64
// in one shot and tripped V8's max single-string length.
//
// Each chunk now lives in its own dedicated scope (mem:idx:vec:NNNN) so
// iii-engine's file-based KV adapter never has to rewrite a multi-chunk
// .bin file. Putting many chunks in a single scope was the failure mode
// at ~500k docs — see follow-up "chunk scope sharding" in the
// persistence memory.
const VECTOR_FORMAT = "chunks-v2";
// Empirically measured 2026-05-23: iii-sdk WS round-trip is fast (<100 ms)
// for state::set values up to ~12 MB, then hard-times out at 16 MB
// (likely a WebSocket frame ceiling somewhere in iii-engine or the
// underlying ws library). 8 MB leaves ample headroom — at ~5 KB JSON
// per vector record that's ~1.6k records/chunk, ~300 chunks for a
// 489k-vector corpus, ~20 s total save wall-clock.
const VECTOR_CHUNK_TARGET_BYTES = 8 * 1024 * 1024;

function vectorChunkScope(index: number): string {
  return `mem:idx:vec:${index.toString().padStart(4, "0")}`;
}

type VectorMeta = {
  format: string;
  count: number;
  chunks: number;
};

// Chunked BM25 persistence — same V8 max-string-length pitfall as the
// vector index, but for the inverted/docTerms tables. Hit live at ~500k
// docs (cold rebuild on 2026-05-22). Three independent chunk streams +
// meta written to the bm25Index scope as the commit point.
const BM25_FORMAT = "chunks-v2";
const BM25_CHUNK_TARGET_BYTES = 8 * 1024 * 1024;

function bm25ChunkScope(kind: "entries" | "inverted" | "docTerms", index: number): string {
  return `mem:idx:bm25:${kind}:${index.toString().padStart(4, "0")}`;
}

type Bm25Meta = {
  format: string;
  count: number;
  entryChunks: number;
  invertedChunks: number;
  docTermsChunks: number;
  totalDocLength: number;
};

export class IndexPersistence {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private lastFailureLogAt = 0;

  constructor(
    private kv: StateKV,
    private bm25: SearchIndex,
    private vector: VectorIndex | null,
  ) {}

  scheduleSave(): void {
    if (this.timer) clearTimeout(this.timer);
    // setTimeout discards the returned promise, so any rejection inside
    // save() would surface as unhandledRejection and crash the process
    // under sustained iii-engine write timeouts (issue #204). Funnel
    // rejections through logFailure() instead.
    this.timer = setTimeout(() => {
      this.save().catch((err) => this.logFailure(err));
    }, DEBOUNCE_MS);
  }

  async save(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    try {
      await this.saveBm25(this.bm25);
      if (this.vector) {
        await this.saveVector(this.vector);
      }
    } catch (err) {
      this.logFailure(err);
    }
  }

  private async saveBm25(bm25: SearchIndex): Promise<void> {
    const prevMeta = await this.kv
      .get<Bm25Meta>(KV.bm25Index, "bm25.meta")
      .catch(() => null);

    if (bm25.size === 0) {
      // Drop everything if the in-memory index emptied.
      for (let i = 0; i < (prevMeta?.entryChunks ?? 0); i++) {
        await this.kv
          .delete(bm25ChunkScope("entries", i), "data")
          .catch(() => {});
      }
      for (let i = 0; i < (prevMeta?.invertedChunks ?? 0); i++) {
        await this.kv
          .delete(bm25ChunkScope("inverted", i), "data")
          .catch(() => {});
      }
      for (let i = 0; i < (prevMeta?.docTermsChunks ?? 0); i++) {
        await this.kv
          .delete(bm25ChunkScope("docTerms", i), "data")
          .catch(() => {});
      }
      await this.kv.delete(KV.bm25Index, "bm25.meta").catch(() => {});
      return;
    }

    const writeStream = async (
      kind: "entries" | "inverted" | "docTerms",
      chunks: IterableIterator<{ index: number; json: string }>,
    ): Promise<number> => {
      let written = 0;
      for (const { index, json } of chunks) {
        await this.kv.set(bm25ChunkScope(kind, index), "data", json);
        written = index + 1;
        // Yield between chunk writes so the REST accept queue keeps
        // draining — same lesson as the vector chunked save.
        await new Promise<void>((resolve) => setImmediate(resolve));
      }
      return written;
    };

    const entryChunks = await writeStream(
      "entries",
      bm25.serializeEntriesChunks(BM25_CHUNK_TARGET_BYTES),
    );
    const invertedChunks = await writeStream(
      "inverted",
      bm25.serializeInvertedChunks(BM25_CHUNK_TARGET_BYTES),
    );
    const docTermsChunks = await writeStream(
      "docTerms",
      bm25.serializeDocTermsChunks(BM25_CHUNK_TARGET_BYTES),
    );

    // Commit point: meta written last. A crash before this leaves the
    // previous chunks readable under their old meta.
    const meta: Bm25Meta = {
      format: BM25_FORMAT,
      count: bm25.size,
      entryChunks,
      invertedChunks,
      docTermsChunks,
      totalDocLength: bm25.totalDocLengthForPersist,
    };
    await this.kv.set(KV.bm25Index, "bm25.meta", meta);

    // Drop orphan chunks from a previously larger save.
    for (let i = entryChunks; i < (prevMeta?.entryChunks ?? 0); i++) {
      await this.kv
        .delete(bm25ChunkScope("entries", i), "data")
        .catch(() => {});
    }
    for (let i = invertedChunks; i < (prevMeta?.invertedChunks ?? 0); i++) {
      await this.kv
        .delete(bm25ChunkScope("inverted", i), "data")
        .catch(() => {});
    }
    for (let i = docTermsChunks; i < (prevMeta?.docTermsChunks ?? 0); i++) {
      await this.kv
        .delete(bm25ChunkScope("docTerms", i), "data")
        .catch(() => {});
    }
  }

  private async saveVector(vector: VectorIndex): Promise<void> {
    const prevMeta = await this.kv
      .get<VectorMeta>(KV.bm25Index, "vectors.meta")
      .catch(() => null);
    const prevChunks = prevMeta?.chunks ?? 0;

    if (vector.size === 0) {
      for (let i = 0; i < prevChunks; i++) {
        await this.kv.delete(vectorChunkScope(i), "data").catch(() => {});
      }
      await this.kv.delete(KV.bm25Index, "vectors.meta").catch(() => {});
      return;
    }

    let written = 0;
    for (const { index, json } of vector.serializeChunks(
      VECTOR_CHUNK_TARGET_BYTES,
    )) {
      await this.kv.set(vectorChunkScope(index), "data", json);
      written = index + 1;
      // Yield to the event loop so the REST accept queue drains between
      // chunks — without this, large saves stall livez and CLOSE_WAITs
      // pile up. See incident artifacts in the linked memory.
      await new Promise<void>((resolve) => setImmediate(resolve));
    }

    // Commit point: meta written last. A crash before this leaves the
    // *previous* meta (and previous chunk set) intact and loadable.
    const meta: VectorMeta = {
      format: VECTOR_FORMAT,
      count: vector.size,
      chunks: written,
    };
    await this.kv.set(KV.bm25Index, "vectors.meta", meta);

    // Drop now-orphan chunks from a previously larger save.
    for (let i = written; i < prevChunks; i++) {
      await this.kv.delete(vectorChunkScope(i), "data").catch(() => {});
    }
  }

  async load(): Promise<{
    bm25: SearchIndex | null;
    vector: VectorIndex | null;
  }> {
    let bm25: SearchIndex | null = null;
    let vector: VectorIndex | null = null;

    const bm25Meta = await this.kv
      .get<Bm25Meta>(KV.bm25Index, "bm25.meta")
      .catch(() => null);
    if (
      bm25Meta &&
      bm25Meta.format === BM25_FORMAT &&
      typeof bm25Meta.entryChunks === "number"
    ) {
      bm25 = new SearchIndex();
      for (let i = 0; i < (bm25Meta.entryChunks ?? 0); i++) {
        const chunkJson = await this.kv
          .get<string>(bm25ChunkScope("entries", i), "data")
          .catch(() => null);
        if (typeof chunkJson === "string") bm25.appendEntriesChunkJson(chunkJson);
      }
      for (let i = 0; i < (bm25Meta.invertedChunks ?? 0); i++) {
        const chunkJson = await this.kv
          .get<string>(bm25ChunkScope("inverted", i), "data")
          .catch(() => null);
        if (typeof chunkJson === "string")
          bm25.appendInvertedChunkJson(chunkJson);
      }
      for (let i = 0; i < (bm25Meta.docTermsChunks ?? 0); i++) {
        const chunkJson = await this.kv
          .get<string>(bm25ChunkScope("docTerms", i), "data")
          .catch(() => null);
        if (typeof chunkJson === "string")
          bm25.appendDocTermsChunkJson(chunkJson);
      }
      bm25.setTotalDocLengthFromLoad(bm25Meta.totalDocLength ?? 0);
    } else {
      // Legacy single-blob format. Loads only if the string fits within
      // V8's max-string-length; at large corpus sizes this throws and we
      // fall through to an empty index + rebuild path.
      const bm25Data = await this.kv
        .get<string>(KV.bm25Index, "data")
        .catch(() => null);
      if (bm25Data && typeof bm25Data === "string") {
        bm25 = SearchIndex.deserialize(bm25Data);
      }
    }

    const meta = await this.kv
      .get<VectorMeta>(KV.bm25Index, "vectors.meta")
      .catch(() => null);
    if (
      meta &&
      meta.format === VECTOR_FORMAT &&
      typeof meta.chunks === "number" &&
      meta.chunks > 0
    ) {
      vector = new VectorIndex();
      for (let i = 0; i < meta.chunks; i++) {
        const chunkJson = await this.kv
          .get<string>(vectorChunkScope(i), "data")
          .catch(() => null);
        if (typeof chunkJson === "string") vector.appendChunkJson(chunkJson);
      }
    } else {
      // Legacy single-blob format. Only loads if it fits within V8's max
      // string length — once persistence runs in the new format we keep
      // the legacy key untouched (upstream's call to clean it up).
      const vecData = await this.kv
        .get<string>(KV.bm25Index, "vectors")
        .catch(() => null);
      if (vecData && typeof vecData === "string") {
        vector = VectorIndex.deserialize(vecData);
      }
    }

    // Final fallback: chunked-load and legacy-blob both produced nothing
    // (corrupt state, fresh install with an existing mem:emb store, or
    // mid-save crash before vectors.meta committed). Rebuild from the
    // per-obs embedding store — no provider re-embed, just KV reads.
    if (!vector || vector.size === 0) {
      const rebuilt = await loadVectorIndexFromStore(this.kv);
      if (rebuilt) {
        vector = rebuilt;
        logger.info(
          "vector index: rebuilt from per-obs embedding store (chunked load was empty)",
          { size: rebuilt.size },
        );
      }
    }

    return { bm25, vector };
  }

  // First-boot migration after upgrade: chunked load gave us an in-memory
  // index, but the per-obs store is empty. Backfill it so the next time
  // the chunked file gets corrupted we can rebuild cheaply. Runs once per
  // process; safe to call on every boot because the empty-store check is
  // a single kv.list.
  async backfillEmbeddingStoreIfEmpty(vector: VectorIndex | null): Promise<void> {
    if (!vector || vector.size === 0) return;
    try {
      if (!(await isEmbeddingStoreEmpty(this.kv))) return;
    } catch (err) {
      logger.warn("embedding-store: probe during migration check failed", {
        error: err instanceof Error ? err.message : String(err),
      });
      return;
    }
    logger.info("embedding-store: backfilling per-obs store from in-memory index", {
      size: vector.size,
    });
    const result = await backfillEmbeddingStoreFromIndex(this.kv, vector);
    logger.info("embedding-store: backfill complete", result);
  }

  stop(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private logFailure(err: unknown): void {
    const now = Date.now();
    // Throttle: persistence failures under load arrive in bursts
    // (iii-engine queue pressure). Logging every debounce flush adds
    // noise without information.
    if (now - this.lastFailureLogAt < FAILURE_LOG_THROTTLE_MS) return;
    this.lastFailureLogAt = now;
    const code = (err as { code?: string })?.code;
    const message = err instanceof Error ? err.message : String(err);
    logger.warn("index persistence: failed to save BM25/vector index", {
      code,
      message,
      hint:
        code === "TIMEOUT"
          ? "iii-engine state::set timed out; recent index updates remain in memory and will retry on the next debounce flush"
          : undefined,
    });
  }
}
