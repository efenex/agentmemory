import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

import { registerApiTriggers } from "../src/triggers/api.js";
import { KV } from "../src/state/schema.js";

const TOTAL = 3000;
const KEYLESS = 46;

type Row = Record<string, unknown>;

function buildStore() {
  const sessions: Row[] = [];
  const summaries: Row[] = [];
  for (let i = 0; i < TOTAL; i++) {
    const id = `ses_${String(i).padStart(5, "0")}`;
    sessions.push({
      id,
      project: "synthetic",
      cwd: "/tmp/synthetic",
      startedAt: new Date(Date.UTC(2026, 0, 1) + i * 60_000).toISOString(),
      status: "completed",
      observationCount: 1,
    });
    if (i % 2 === 0) {
      summaries.push({ sessionId: id, project: "synthetic", title: `t${i}` });
    }
  }
  for (let i = 0; i < KEYLESS; i++) {
    sessions.push({ project: "synthetic", startedAt: "2026-01-01T00:00:00Z" });
  }
  return { sessions, summaries };
}

function setup() {
  const { sessions, summaries } = buildStore();
  const calls = { get: [] as Array<[string, unknown]>, list: [] as string[] };
  const kv = {
    get: async (scope: string, key: unknown) => {
      calls.get.push([scope, key]);
      if (typeof key !== "string") throw new Error("missing field 'key'");
      return (
        summaries.find((s) => scope === KV.summaries && s.sessionId === key) ??
        null
      );
    },
    list: async (scope: string) => {
      calls.list.push(scope);
      if (scope === KV.sessions) return [...sessions];
      if (scope === KV.summaries) return [...summaries];
      return [];
    },
    set: async () => {},
    delete: async () => {},
  };
  const fns = new Map<string, Function>();
  const sdk = {
    registerFunction: (id: string, h: Function) => fns.set(id, h),
    registerTrigger: () => {},
    trigger: async () => undefined,
  };
  registerApiTriggers(sdk as never, kv as never);
  const call = async (query: Record<string, string> = {}) => {
    calls.get.length = 0;
    calls.list.length = 0;
    const res = await fns.get("api::sessions")!({ headers: {}, query_params: query });
    return res as { status_code: number; body: any };
  };
  return { call, calls };
}

describe("GET /sessions limit/offset", () => {
  let ctx: ReturnType<typeof setup>;
  beforeEach(() => {
    ctx = setup();
  });

  it("limit=1 fetches exactly one summary instead of one per session", async () => {
    const res = await ctx.call({ limit: "1" });
    expect(res.status_code).toBe(200);
    expect(res.body.sessions).toHaveLength(1);
    expect(ctx.calls.get.filter(([s]) => s === KV.summaries)).toHaveLength(1);
    expect(ctx.calls.list.filter((s) => s === KV.summaries)).toHaveLength(0);
  });

  it("returns the newest sessions first, matching replay::sessions order", async () => {
    const res = await ctx.call({ limit: "3" });
    expect(res.body.sessions.map((s: Row) => s.id)).toEqual([
      "ses_02999",
      "ses_02998",
      "ses_02997",
    ]);
    expect(res.body.total).toBe(TOTAL);
  });

  it("attaches summaries only where one exists", async () => {
    const res = await ctx.call({ limit: "2" });
    const [newest, next] = res.body.sessions;
    expect(newest.summary).toBeUndefined();
    expect(next.summary?.sessionId).toBe("ses_02998");
  });

  it("offset pages without overlap", async () => {
    const a = await ctx.call({ limit: "5", offset: "0" });
    const b = await ctx.call({ limit: "5", offset: "5" });
    const ids = [...a.body.sessions, ...b.body.sessions].map((s: Row) => s.id);
    expect(new Set(ids).size).toBe(10);
    expect(b.body.sessions[0].id).toBe("ses_02994");
    expect(b.body.offset).toBe(5);
  });

  it("offset past the end returns an empty page without summary gets", async () => {
    const res = await ctx.call({ limit: "10", offset: "999999" });
    expect(res.body.sessions).toEqual([]);
    expect(ctx.calls.get).toHaveLength(0);
  });

  it("offset alone paginates without a limit cap", async () => {
    const res = await ctx.call({ offset: "2990" });
    expect(res.body.sessions).toHaveLength(10);
  });

  it("clamps limit below 1 to 1", async () => {
    const res = await ctx.call({ limit: "0" });
    expect(res.body.sessions).toHaveLength(1);
  });

  it("never fetches a summary for a keyless record and excludes them from pages", async () => {
    const res = await ctx.call({ limit: "5000" });
    expect(ctx.calls.get.every(([, key]) => typeof key === "string")).toBe(true);
    expect(res.body.sessions).toHaveLength(TOTAL);
    expect(res.body.sessions.every((s: Row) => typeof s.id === "string")).toBe(true);
    expect(res.body.total).toBe(TOTAL);
  });

  it("ignores a non-numeric limit and keeps the unbounded path", async () => {
    const res = await ctx.call({ limit: "abc" });
    expect(res.body.sessions).toHaveLength(TOTAL + KEYLESS);
  });

  it("unbounded path: one summaries list, no per-session gets, same response shape", async () => {
    const res = await ctx.call();
    expect(res.status_code).toBe(200);
    expect(Object.keys(res.body)).toEqual(["sessions"]);
    expect(res.body.sessions).toHaveLength(TOTAL + KEYLESS);
    expect(ctx.calls.get).toHaveLength(0);
    expect(ctx.calls.list.filter((s) => s === KV.summaries)).toHaveLength(1);
    const withSummary = res.body.sessions.filter((s: Row) => s.summary);
    expect(withSummary).toHaveLength(TOTAL / 2);
    const first = res.body.sessions[0];
    expect(first.summary.sessionId).toBe(first.id);
  });
});
