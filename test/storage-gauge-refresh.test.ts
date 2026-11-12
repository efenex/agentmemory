import { describe, it, expect, beforeEach, vi } from "vitest";

const logger = vi.hoisted(() => ({
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
}));
vi.mock("../src/logger.js", () => ({ logger }));

import {
  memoriesTotal,
  sessionsTotal,
  lessonsTotal,
  refreshStorageGauges,
  resetStorageGaugeFailureState,
} from "../src/telemetry/prometheus.js";
import {
  resetColdScanBackoff,
  coldScanBackedOff,
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

type ListImpl = (scope: string) => Promise<unknown[]>;

function kvWith(list: ListImpl, get?: (scope: string, key: string) => Promise<unknown>) {
  return {
    list: vi.fn(list),
    get: vi.fn(get ?? (async () => null)),
  };
}

async function memorySamples() {
  const { values } = await memoriesTotal.get();
  return Object.fromEntries(values.map((v) => [v.labels.type as string, v.value]));
}

const okLists: ListImpl = async (scope) => {
  if (scope === "mem:memories") {
    return [
      makeMemory("mem_1", { type: "fact" }),
      makeMemory("mem_2", { type: "fact" }),
      makeMemory("mem_3", { type: "pattern" }),
      makeMemory("mem_4", { type: "fact", isLatest: false }),
    ];
  }
  return [];
};

describe("refreshStorageGauges (full-list path, index not ready)", () => {
  beforeEach(() => {
    memoriesTotal.reset();
    sessionsTotal.reset();
    lessonsTotal.reset();
    resetStorageGaugeFailureState();
    resetColdScanBackoff();
    logger.info.mockClear();
    logger.warn.mockClear();
  });

  it("counts latest memories by type", async () => {
    await refreshStorageGauges(kvWith(okLists) as never, undefined);
    expect(await memorySamples()).toEqual({ fact: 2, pattern: 1 });
  });

  it("keeps the last good series when the memories list later fails", async () => {
    await refreshStorageGauges(kvWith(okLists) as never, undefined);
    const failing = kvWith(async (scope) => {
      if (scope === "mem:memories") throw new Error("Invocation timeout after 180000ms: state::list");
      return [];
    });
    await refreshStorageGauges(failing as never, undefined);
    expect(await memorySamples()).toEqual({ fact: 2, pattern: 1 });
  });

  it("does not reset or emit a series when the first list fails", async () => {
    const resetSpy = vi.spyOn(memoriesTotal, "reset");
    const failing = kvWith(async (scope) => {
      if (scope === "mem:memories") throw new Error("timeout");
      return [];
    });
    await refreshStorageGauges(failing as never, undefined);
    expect(resetSpy).not.toHaveBeenCalled();
    expect(await memorySamples()).toEqual({});
    resetSpy.mockRestore();
  });

  it("still clears the gauge when a successful list is empty", async () => {
    await refreshStorageGauges(kvWith(okLists) as never, undefined);
    await refreshStorageGauges(kvWith(async () => []) as never, undefined);
    expect(await memorySamples()).toEqual({});
  });

  it("logs a failing scope once per outage and logs recovery", async () => {
    const failing = kvWith(async (scope) => {
      if (scope === "mem:memories") throw new Error("timeout");
      return [];
    });
    await refreshStorageGauges(failing as never, undefined);
    resetColdScanBackoff();
    await refreshStorageGauges(failing as never, undefined);
    const warns = logger.warn.mock.calls.filter(
      (c) => c[0] === "storage gauge: list failed, keeping last value",
    );
    expect(warns).toHaveLength(1);
    expect(warns[0][1]).toMatchObject({ scope: "memories" });

    resetColdScanBackoff();
    await refreshStorageGauges(kvWith(okLists) as never, undefined);
    expect(logger.info).toHaveBeenCalledWith("storage gauge: list recovered", {
      scope: "memories",
    });
  });

  it("skips the full list while backed off after a failed scan", async () => {
    const failing = kvWith(async (scope) => {
      if (scope === "mem:memories") throw new Error("timeout");
      return [];
    });
    await refreshStorageGauges(failing as never, undefined);
    expect(coldScanBackedOff()).toBe(true);
    failing.list.mockClear();
    await refreshStorageGauges(failing as never, undefined);
    expect(failing.list.mock.calls.map((c) => c[0])).not.toContain("mem:memories");
  });

  it("keeps other gauges updating when only the memories list fails", async () => {
    const kv = kvWith(async (scope) => {
      if (scope === "mem:memories") throw new Error("timeout");
      if (scope === "mem:lessons") return [{ id: "l1" }, { id: "l2", deleted: true }];
      return [];
    });
    await refreshStorageGauges(kv as never, undefined);
    expect((await lessonsTotal.get()).values[0].value).toBe(1);
  });
});

describe("refreshStorageGauges (index path)", () => {
  beforeEach(() => {
    memoriesTotal.reset();
    resetStorageGaugeFailureState();
    resetColdScanBackoff();
    logger.warn.mockClear();
  });

  it("counts from index ids with kv.get and never lists the memories scope", async () => {
    markMemoryIndexReady();
    const idx = getSearchIndex();
    idx.clear();
    const rows: Record<string, Memory> = {
      mem_a: makeMemory("mem_a", { type: "fact" }),
      mem_b: makeMemory("mem_b", { type: "pattern" }),
    };
    for (const m of Object.values(rows)) idx.add(memoryToObservation(m));
    const kv = kvWith(async () => [], async (_s, key) => rows[key] ?? null);

    await refreshStorageGauges(kv as never, undefined);

    expect(await memorySamples()).toEqual({ fact: 1, pattern: 1 });
    expect(kv.list.mock.calls.map((c) => c[0])).not.toContain("mem:memories");
    idx.clear();
  });

  it("keeps the last value when an index-path kv.get fails", async () => {
    markMemoryIndexReady();
    const idx = getSearchIndex();
    idx.clear();
    memoriesTotal.set({ type: "fact" }, 7);
    idx.add(memoryToObservation(makeMemory("mem_a")));
    const kv = kvWith(async () => [], async () => {
      throw new Error("boom");
    });
    await refreshStorageGauges(kv as never, undefined);
    expect(await memorySamples()).toEqual({ fact: 7 });
    idx.clear();
  });
});
