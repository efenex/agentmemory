import type { Memory } from "../types.js";
import { KV } from "./schema.js";
import type { StateKV } from "./kv.js";
import { getSearchIndex, isMemoryIndexReady } from "../functions/search.js";

const COLD_SCAN_BACKOFF_MS = 10 * 60 * 1000;
const GET_BATCH_SIZE = 50;
const MEMORY_ID_PREFIX = "mem_";

let coldScanBackoffUntil = 0;

export function coldScanBackedOff(): boolean {
  return Date.now() < coldScanBackoffUntil;
}

export function noteColdScanFailure(): void {
  coldScanBackoffUntil = Date.now() + COLD_SCAN_BACKOFF_MS;
}

export function resetColdScanBackoff(): void {
  coldScanBackoffUntil = 0;
}

/**
 * Enumerate memories without a full-scope state::list, which on a large
 * file-backed store runs past the 180s engine invocation timeout and pins
 * the engine. Walks the BM25 index's mem_ ids and loads them with bounded
 * parallel kv.get batches.
 *
 * Returns null when the index is not ready or holds no mem_ ids (no
 * cheap path available; callers must not treat that as "zero memories").
 * Memories the index does not cover (not yet backfilled, or
 * superseded revisions) are not returned, so the result is a lower bound
 * on the memories scope. Stale index ids resolve to null and are skipped.
 */
export async function listMemoriesViaIndex(
  kv: StateKV,
): Promise<Memory[] | null> {
  const idx = getSearchIndex();
  if (!isMemoryIndexReady() || idx.size === 0) return null;
  const ids = idx.ids().filter((id) => id.startsWith(MEMORY_ID_PREFIX));
  if (ids.length === 0) return null;
  const memories: Memory[] = [];
  for (let start = 0; start < ids.length; start += GET_BATCH_SIZE) {
    const loaded = await Promise.all(
      ids
        .slice(start, start + GET_BATCH_SIZE)
        .map((id) => kv.get<Memory>(KV.memories, id)),
    );
    for (const mem of loaded) if (mem) memories.push(mem);
  }
  return memories;
}
