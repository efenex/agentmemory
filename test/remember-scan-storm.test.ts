import { describe, it, expect, vi, beforeEach } from "vitest";
import { mockKV, mockSdk } from "./helpers/mocks.js";

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

async function setup() {
  vi.resetModules();
  const search = await import("../src/functions/search.js");
  const { registerRememberFunction } = await import("../src/functions/remember.js");
  const sdk = mockSdk({ looseTrigger: true });
  const kv = mockKV();
  registerRememberFunction(sdk as never, kv as never);
  return { sdk, kv, search };
}

function memoryListCalls(kv: { list: unknown }): number {
  return (kv.list as ReturnType<typeof vi.fn>).mock.calls.filter(
    (c) => c[0] === "mem:memories",
  ).length;
}

describe("mem::remember candidate lookup under a failing scan", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it("saves once without a second scan when the cold-index scan fails", async () => {
    const { sdk, kv } = await setup();
    const list = vi.spyOn(kv, "list").mockRejectedValue(
      new Error("Invocation timeout after 180000ms: state::list"),
    );

    const res = (await sdk.trigger("mem::remember", {
      content: "scan failure must not block the save itself",
    })) as { success: boolean; memory: { id: string } };

    expect(res.success).toBe(true);
    expect(res.memory.id).toMatch(/^mem_/);
    expect(list).toHaveBeenCalledTimes(1);
  });

  it("skips cold scans for a while after one fails", async () => {
    const { sdk, kv } = await setup();
    const list = vi.spyOn(kv, "list").mockRejectedValue(new Error("timeout"));

    await sdk.trigger("mem::remember", { content: "first save fails the scan" });
    await sdk.trigger("mem::remember", { content: "second save skips the scan" });
    await sdk.trigger("mem::remember", { content: "third save skips the scan too" });

    expect(list).toHaveBeenCalledTimes(1);
  });

  it("never scans the memories scope once the index is marked ready", async () => {
    const { sdk, kv, search } = await setup();
    await sdk.trigger("mem::remember", { content: "seed memory for the index" });
    search.markMemoryIndexReady();
    const list = vi.spyOn(kv, "list");

    const before = memoryListCalls(kv);
    await sdk.trigger("mem::remember", { content: "another memory after the index is ready" });
    expect(memoryListCalls(kv)).toBe(before);
    expect(list).not.toHaveBeenCalled();
  });
});
