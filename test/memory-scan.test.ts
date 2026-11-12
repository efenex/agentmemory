import { describe, it, expect, beforeEach, vi } from "vitest";

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

import {
  coldScanBackedOff,
  listMemoriesViaIndex,
  noteColdScanFailure,
  resetColdScanBackoff,
} from "../src/state/memory-scan.js";
import {
  getSearchIndex,
  markMemoryIndexReady,
} from "../src/functions/search.js";
import { memoryToObservation } from "../src/state/memory-utils.js";
import type { Memory } from "../src/types.js";

function makeMemory(id: string, overrides: Partial<Memory> = {}): Memory {
  return {
    id,
    createdAt: "2026-01-01T00:00:00Z",
    updatedAt: "2026-01-01T00:00:00Z",
    type: "fact",
    title: `title ${id}`,
    content: `content for ${id}`,
    concepts: [],
    files: [],
    sessionIds: [],
    strength: 5,
    version: 1,
    isLatest: true,
    ...overrides,
  };
}

function fakeKv(rows: Record<string, Memory>) {
  const get = vi.fn(async (_scope: string, key: string) => rows[key] ?? null);
  const list = vi.fn(async () => Object.values(rows));
  return { get, list, kv: { get, list } as never };
}

describe("memory-scan cold-scan backoff", () => {
  beforeEach(() => resetColdScanBackoff());

  it("is inactive until a failure is noted, then active, then resettable", () => {
    expect(coldScanBackedOff()).toBe(false);
    noteColdScanFailure();
    expect(coldScanBackedOff()).toBe(true);
    resetColdScanBackoff();
    expect(coldScanBackedOff()).toBe(false);
  });
});

describe("listMemoriesViaIndex", () => {
  it("returns null while the memory index is not ready", async () => {
    const { kv, get, list } = fakeKv({});
    expect(await listMemoriesViaIndex(kv)).toBeNull();
    expect(get).not.toHaveBeenCalled();
    expect(list).not.toHaveBeenCalled();
  });

  it("returns null when ready but the index is empty", async () => {
    markMemoryIndexReady();
    getSearchIndex().clear();
    const { kv } = fakeKv({});
    expect(await listMemoriesViaIndex(kv)).toBeNull();
  });

  it("loads mem_ ids via bounded kv.get batches, skipping stale and non-memory ids, never kv.list", async () => {
    markMemoryIndexReady();
    const idx = getSearchIndex();
    idx.clear();
    const rows: Record<string, Memory> = {};
    for (let i = 0; i < 120; i++) {
      const m = makeMemory(`mem_${i}`);
      rows[m.id] = m;
      idx.add(memoryToObservation(m));
    }
    idx.add(memoryToObservation(makeMemory("mem_stale")));
    idx.add({
      id: "obs_1",
      sessionId: "ses_1",
      timestamp: "2026-01-01T00:00:00Z",
      type: "other",
      title: "an observation",
      facts: [],
      narrative: "n",
      concepts: [],
      files: [],
      importance: 5,
    } as never);

    const { kv, get, list } = fakeKv(rows);
    const out = await listMemoriesViaIndex(kv);

    expect(out).toHaveLength(120);
    expect(get).toHaveBeenCalledTimes(121);
    expect(get.mock.calls.every((c) => String(c[1]).startsWith("mem_"))).toBe(true);
    expect(list).not.toHaveBeenCalled();
    idx.clear();
  });

  it("propagates a kv.get failure to the caller", async () => {
    markMemoryIndexReady();
    const idx = getSearchIndex();
    idx.clear();
    idx.add(memoryToObservation(makeMemory("mem_a")));
    const kv = {
      get: vi.fn(async () => {
        throw new Error("boom");
      }),
    } as never;
    await expect(listMemoriesViaIndex(kv)).rejects.toThrow("boom");
    idx.clear();
  });
});
