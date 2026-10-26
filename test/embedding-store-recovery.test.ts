import { describe, it, expect, vi } from "vitest";
import { backfillEmbeddingStoreFromIndex, loadVectorIndexFromStore, persistEmbedding } from "../src/state/embedding-store.js";
import { IndexPersistence } from "../src/state/index-persistence.js";
import { VectorIndex } from "../src/state/vector-index.js";

function store() {
  const scopes = new Map<string, Map<string, unknown>>();
  return {
    get: vi.fn(async (scope: string, id: string) => scopes.get(scope)?.get(id) ?? null),
    set: vi.fn(async (scope: string, id: string, value: unknown) => {
      if (!scopes.has(scope)) scopes.set(scope, new Map());
      scopes.get(scope)!.set(id, value);
      return value;
    }),
    list: vi.fn(async (scope: string) => [...(scopes.get(scope)?.values() ?? [])]),
    delete: vi.fn(async (scope: string, id: string) => { scopes.get(scope)?.delete(id); }),
  };
}

describe("fork embedding recovery", () => {
  it("backfills quantized vectors using the public float-vector iterator", async () => {
    const kv = store();
    const vector = new VectorIndex();
    vector.add("obs_aa", "ses_a", new Float32Array([1, 2, 3]));
    expect(await backfillEmbeddingStoreFromIndex(kv as never, vector)).toEqual({ written: 1, failed: 0 });
    const loaded = await loadVectorIndexFromStore(kv as never);
    expect(loaded?.size).toBe(1);
    expect(loaded?.search(new Float32Array([1, 2, 3]))[0].score).toBeCloseTo(1, 2);
  });

  it("recovers stored embeddings when no vector snapshot remains", async () => {
    const kv = store();
    await persistEmbedding(kv as never, "obs_aa", "ses_a", new Float32Array([1, 0, 0]));
    const persistence = new IndexPersistence(kv as never, new VectorIndex(), { requireRecoveryPlan: true });
    kv.set.mockClear();
    const loaded = await persistence.load();
    expect(loaded.state).toBe("migrated");
    expect(loaded.vector?.has("obs_aa")).toBe(true);
    expect(loaded.vector?.pendingChanges).toBe(1);
    expect(kv.set).not.toHaveBeenCalled();
    persistence.stop();
  });

  it("does not certify a partially read embedding store as a recoverable checkpoint", async () => {
    const kv = store();
    await persistEmbedding(kv as never, "obs_aa", "ses_a", new Float32Array([1, 0, 0]));
    const list = kv.list.getMockImplementation()!;
    kv.list.mockImplementation(async (scope) => {
      if (scope === "mem:emb:31") throw new Error("state::list timeout");
      return list(scope);
    });
    const persistence = new IndexPersistence(kv as never, new VectorIndex(), { requireRecoveryPlan: true });
    const loaded = await persistence.load();
    expect(loaded.state).toBe("unavailable");
    await persistence.prepareBackfill(loaded);
    expect(persistence.isRecoveryPaused()).toBe(true);
    persistence.stop();
  });
});
