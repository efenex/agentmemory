import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { IndexPersistence } from "../src/state/index-persistence.js";
import { SearchIndex } from "../src/state/search-index.js";
import { VectorIndex } from "../src/state/vector-index.js";
import type { CompressedObservation } from "../src/types.js";

function mockKV() {
  const store = new Map<string, Map<string, unknown>>();
  return {
    get: async <T>(scope: string, key: string): Promise<T | null> => {
      return (store.get(scope)?.get(key) as T) ?? null;
    },
    set: async <T>(scope: string, key: string, data: T): Promise<T> => {
      if (!store.has(scope)) store.set(scope, new Map());
      store.get(scope)!.set(key, data);
      return data;
    },
    delete: async (scope: string, key: string): Promise<void> => {
      store.get(scope)?.delete(key);
    },
    list: async <T>(scope: string): Promise<T[]> => {
      const entries = store.get(scope);
      return entries ? (Array.from(entries.values()) as T[]) : [];
    },
  };
}

function makeObs(
  overrides: Partial<CompressedObservation> = {},
): CompressedObservation {
  return {
    id: "obs_1",
    sessionId: "ses_1",
    timestamp: new Date().toISOString(),
    type: "file_edit",
    title: "Edit auth middleware",
    subtitle: "JWT validation",
    facts: ["Added token check"],
    narrative: "Modified the auth middleware to validate JWT tokens",
    concepts: ["authentication", "jwt"],
    files: ["src/middleware/auth.ts"],
    importance: 7,
    ...overrides,
  };
}

describe("IndexPersistence", () => {
  let kv: ReturnType<typeof mockKV>;

  beforeEach(() => {
    vi.useFakeTimers();
    kv = mockKV();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("saves and loads BM25 index round-trip", async () => {
    // Fork (chunks-v2, audit F10a): save() yields to the event loop via
    // setImmediate between chunk writes so a large save doesn't stall the
    // REST accept queue. vitest's fake timers also fake setImmediate, which
    // would stall save() forever; this test doesn't exercise timing, so run
    // on real timers. Upstream's single-blob save has no such yield.
    vi.useRealTimers();
    const bm25 = new SearchIndex();
    bm25.add(makeObs({ id: "obs_1", title: "auth handler" }));

    const persistence = new IndexPersistence(kv as never, bm25, null);
    await persistence.save();

    const loaded = await persistence.load();
    expect(loaded.bm25).not.toBeNull();
    expect(loaded.bm25!.size).toBe(1);
    const results = loaded.bm25!.search("auth");
    expect(results.length).toBe(1);
  });

  it("saves and loads vector index round-trip", async () => {
    // Fork (chunks-v2, audit F10a): save() yields to the event loop via
    // setImmediate between chunk writes so a large save doesn't stall the
    // REST accept queue. vitest's fake timers also fake setImmediate, which
    // would stall save() forever; this test doesn't exercise timing, so run
    // on real timers. Upstream's single-blob save has no such yield.
    vi.useRealTimers();
    const bm25 = new SearchIndex();
    const vector = new VectorIndex();
    vector.add("obs_1", "ses_1", new Float32Array([0.1, 0.2, 0.3]));

    const persistence = new IndexPersistence(kv as never, bm25, vector);
    await persistence.save();

    const loaded = await persistence.load();
    expect(loaded.vector).not.toBeNull();
    expect(loaded.vector!.size).toBe(1);
  });

  it("scheduleSave debounces multiple calls", async () => {
    // Fork (chunks-v2, audit F10a) differs from upstream in two ways:
    // - the commit point is `bm25.meta` in mem:index:bm25 (chunks live in
    //   mem:idx:bm25:*), not the legacy single-blob `data` key;
    // - an empty index is never persisted (save drops chunks + meta), so
    //   the index needs a document for the save to be observable.
    const bm25 = new SearchIndex();
    bm25.add(makeObs({ id: "obs_1", title: "auth handler" }));
    const persistence = new IndexPersistence(kv as never, bm25, null);
    const setSpy = vi.spyOn(kv, "set");

    persistence.scheduleSave();
    persistence.scheduleSave();
    persistence.scheduleSave();

    await expect(kv.get("mem:index:bm25", "bm25.meta")).resolves.toBeNull();
    expect(setSpy).not.toHaveBeenCalled();

    vi.advanceTimersByTime(5000);
    await vi.runAllTimersAsync();

    const meta = await kv.get<{ format: string; count: number }>(
      "mem:index:bm25",
      "bm25.meta",
    );
    expect(meta).not.toBeNull();
    expect(meta!.format).toBe("chunks-v2");
    expect(meta!.count).toBe(1);
    // Three scheduleSave() calls collapse into a single save: exactly one
    // meta commit.
    const metaWrites = setSpy.mock.calls.filter(
      ([scope, key]) => scope === "mem:index:bm25" && key === "bm25.meta",
    );
    expect(metaWrites.length).toBe(1);
  });

  it("stop clears the pending timer", async () => {
    const bm25 = new SearchIndex();
    bm25.add(makeObs({ id: "obs_1", title: "auth handler" }));
    const persistence = new IndexPersistence(kv as never, bm25, null);

    persistence.scheduleSave();
    persistence.stop();

    vi.advanceTimersByTime(10000);
    const saved = await kv.get<string>("mem:index:bm25", "data");
    expect(saved).toBeNull();
  });

  it("returns null indexes when nothing has been saved", async () => {
    const bm25 = new SearchIndex();
    const persistence = new IndexPersistence(kv as never, bm25, null);

    const loaded = await persistence.load();
    expect(loaded.bm25).toBeNull();
    expect(loaded.vector).toBeNull();
  });

  it("scheduled save swallows kv.set rejection without unhandledRejection (#204)", async () => {
    const failingKv = {
      ...mockKV(),
      set: vi.fn(async () => {
        const err = new Error(
          "TIMEOUT: invocation timed out after 30000ms",
        ) as Error & { code?: string; function_id?: string };
        err.code = "TIMEOUT";
        err.function_id = "state::set";
        throw err;
      }),
    };
    const bm25 = new SearchIndex();
    bm25.add(makeObs({ id: "obs_1", title: "auth handler" }));
    const persistence = new IndexPersistence(failingKv as never, bm25, null);

    let unhandled = false;
    const onUnhandled = () => {
      unhandled = true;
    };
    process.on("unhandledRejection", onUnhandled);

    try {
      persistence.scheduleSave();
      vi.advanceTimersByTime(5000);
      await vi.runAllTimersAsync();
      // give microtasks a chance to flush
      await Promise.resolve();
      expect(failingKv.set).toHaveBeenCalled();
      expect(unhandled).toBe(false);
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  it("save() does not throw when kv.set rejects (#204)", async () => {
    const failingKv = {
      ...mockKV(),
      set: vi.fn(async () => {
        throw new Error("TIMEOUT");
      }),
    };
    const bm25 = new SearchIndex();
    bm25.add(makeObs({ id: "obs_1", title: "auth handler" }));
    const persistence = new IndexPersistence(failingKv as never, bm25, null);

    await expect(persistence.save()).resolves.toBeUndefined();
  });
});
