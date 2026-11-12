import { describe, it, expect, beforeEach, vi } from "vitest";

const logger = vi.hoisted(() => ({
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
}));
vi.mock("../src/logger.js", () => ({ logger }));

import { registerAutoForgetFunction } from "../src/functions/auto-forget.js";
import {
  coldScanBackedOff,
  resetColdScanBackoff,
} from "../src/state/memory-scan.js";
import {
  getSearchIndex,
  markMemoryIndexReady,
  setIndexPersistence,
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

function setup(memories: Memory[], opts: { failMemoriesList?: boolean } = {}) {
  const store = new Map(memories.map((m) => [m.id, m]));
  const kv = {
    get: vi.fn(async (scope: string, key: string) =>
      scope === "mem:memories" ? (store.get(key) ?? null) : null,
    ),
    set: vi.fn(async (_s: string, _k: string, v: unknown) => v),
    delete: vi.fn(async (scope: string, key: string) => {
      if (scope === "mem:memories") store.delete(key);
    }),
    list: vi.fn(async (scope: string) => {
      if (scope === "mem:memories") {
        if (opts.failMemoriesList) throw new Error("Invocation timeout after 180000ms: state::list");
        return Array.from(store.values());
      }
      return [];
    }),
  };
  const fns = new Map<string, Function>();
  const sdk = {
    registerFunction: (id: string, h: Function) => fns.set(id, h),
    registerTrigger: () => {},
    trigger: async (input: { function_id: string; payload: unknown }) =>
      fns.get(input.function_id)!(input.payload),
  };
  registerAutoForgetFunction(sdk as never, kv as never);
  const run = (payload: Record<string, unknown> = {}) =>
    sdk.trigger({ function_id: "mem::auto-forget", payload }) as Promise<{
      ttlExpired: string[];
    }>;
  return { kv, run };
}

describe("auto-forget memories scan (full-list path, index not ready)", () => {
  beforeEach(() => {
    resetColdScanBackoff();
    logger.warn.mockClear();
  });

  it("rethrows a failed scan, logs it, and starts the cold-scan backoff", async () => {
    const { run } = setup([], { failMemoriesList: true });
    await expect(run({ scheduled: true })).rejects.toThrow(/timeout/);
    expect(logger.warn).toHaveBeenCalledWith(
      "auto-forget memories scan failed",
      expect.objectContaining({ error: expect.stringContaining("timeout") }),
    );
    expect(coldScanBackedOff()).toBe(true);
  });

  it("scheduled runs skip the full scan while backed off", async () => {
    const { kv, run } = setup([makeMemory("mem_1", { forgetAfter: "2020-01-01T00:00:00Z" })]);
    const { noteColdScanFailure } = await import("../src/state/memory-scan.js");
    noteColdScanFailure();

    const result = await run({ scheduled: true });

    expect(result.ttlExpired).toEqual([]);
    expect(kv.list.mock.calls.map((c) => c[0])).not.toContain("mem:memories");
  });

  it("manual runs bypass the backoff", async () => {
    const { run } = setup([makeMemory("mem_1", { forgetAfter: "2020-01-01T00:00:00Z" })]);
    const { noteColdScanFailure } = await import("../src/state/memory-scan.js");
    noteColdScanFailure();

    const result = await run({});

    expect(result.ttlExpired).toEqual(["mem_1"]);
  });
});

describe("auto-forget memories scan (index path)", () => {
  beforeEach(() => {
    resetColdScanBackoff();
    setIndexPersistence(null as never);
  });

  it("expires TTL memories found via index ids without listing the memories scope", async () => {
    markMemoryIndexReady();
    const idx = getSearchIndex();
    idx.clear();
    const expired = makeMemory("mem_exp", { forgetAfter: "2020-01-01T00:00:00Z" });
    const live = makeMemory("mem_live");
    for (const m of [expired, live]) idx.add(memoryToObservation(m));
    const { kv, run } = setup([expired, live], { failMemoriesList: true });

    const result = await run({ scheduled: true });

    expect(result.ttlExpired).toEqual(["mem_exp"]);
    expect(kv.list.mock.calls.map((c) => c[0])).not.toContain("mem:memories");
    expect(kv.get).toHaveBeenCalledWith("mem:memories", "mem_live");
    idx.clear();
  });
});
