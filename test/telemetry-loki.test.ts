import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

// We exercise the *real* logger.ts here (no mock of "../src/logger.js"),
// since the Loki path is implemented inside emit(). Each test resets
// the module so its module-level queue/timer state starts clean.

const ORIGINAL_ENV = { ...process.env };
let fetchMock: ReturnType<typeof vi.fn>;
let originalFetch: typeof globalThis.fetch;

beforeEach(() => {
  vi.resetModules();
  process.env = { ...ORIGINAL_ENV };
  delete process.env["LOKI_PUSH_URL"];
  delete process.env["LOKI_LEVEL"];
  delete process.env["LOKI_BATCH_SIZE"];
  delete process.env["LOKI_BATCH_TIMEOUT_MS"];
  delete process.env["PROMETHEUS_USER"];
  delete process.env["PROMETHEUS_PASSWORD"];

  fetchMock = vi.fn(async () =>
    new Response(null, { status: 204 }),
  );
  originalFetch = globalThis.fetch;
  globalThis.fetch = fetchMock as unknown as typeof globalThis.fetch;
});

afterEach(() => {
  process.env = { ...ORIGINAL_ENV };
  globalThis.fetch = originalFetch;
});

describe("logger Loki push — no-op when disabled", () => {
  it("does not call fetch when LOKI_PUSH_URL is unset", async () => {
    const { logger, flushLokiForTests } = await import("../src/logger.js");
    logger.info("hello", { k: "v" });
    logger.warn("be careful");
    await flushLokiForTests();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("logger Loki push — batching behavior", () => {
  it("flushes when the queue hits LOKI_BATCH_SIZE", async () => {
    process.env["LOKI_PUSH_URL"] = "https://loki.example/loki/api/v1/push";
    process.env["LOKI_BATCH_SIZE"] = "3";
    const { logger } = await import("../src/logger.js");

    logger.info("a");
    logger.info("b");
    expect(fetchMock).not.toHaveBeenCalled();
    logger.info("c"); // hits batch size — flush triggered

    // flush is async; wait a tick for the microtask + fetch
    await new Promise((r) => setTimeout(r, 10));
    expect(fetchMock).toHaveBeenCalledOnce();

    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe("https://loki.example/loki/api/v1/push");
    expect(opts.method).toBe("POST");
    const body = JSON.parse(opts.body);
    expect(body.streams).toHaveLength(1);
    expect(body.streams[0].stream.service).toBe("agentmemory");
    expect(body.streams[0].values).toHaveLength(3);
    // Each entry is [tsNs, jsonLine]
    const line = JSON.parse(body.streams[0].values[0][1]);
    expect(line.msg).toBe("a");
    expect(line.level).toBe("info");
  });

  it("flushes after LOKI_BATCH_TIMEOUT_MS even with no batch hit", async () => {
    vi.useFakeTimers();
    process.env["LOKI_PUSH_URL"] = "https://loki.example/loki/api/v1/push";
    process.env["LOKI_BATCH_SIZE"] = "100";
    process.env["LOKI_BATCH_TIMEOUT_MS"] = "200";
    const { logger } = await import("../src/logger.js");

    logger.info("first line");
    expect(fetchMock).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(250);
    // Switch back to real timers so the fetch mock's microtask runs
    vi.useRealTimers();
    await new Promise((r) => setTimeout(r, 10));

    expect(fetchMock).toHaveBeenCalledOnce();
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.streams[0].values).toHaveLength(1);
  });
});

describe("logger Loki push — filtering + correlation", () => {
  it("LOKI_LEVEL=warn drops info+debug lines", async () => {
    process.env["LOKI_PUSH_URL"] = "https://loki.example/loki/api/v1/push";
    process.env["LOKI_LEVEL"] = "warn";
    process.env["LOKI_BATCH_SIZE"] = "1"; // force immediate flush per line
    const { logger } = await import("../src/logger.js");

    logger.info("should be dropped");
    await new Promise((r) => setTimeout(r, 5));
    expect(fetchMock).not.toHaveBeenCalled();

    logger.warn("should be kept");
    await new Promise((r) => setTimeout(r, 10));
    expect(fetchMock).toHaveBeenCalledOnce();
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    const line = JSON.parse(body.streams[0].values[0][1]);
    expect(line.msg).toBe("should be kept");
    expect(line.level).toBe("warn");
  });

  it("sends Authorization: Basic when user+pass are set", async () => {
    process.env["LOKI_PUSH_URL"] = "https://loki.example/loki/api/v1/push";
    process.env["LOKI_BATCH_SIZE"] = "1";
    process.env["PROMETHEUS_USER"] = "alice";
    process.env["PROMETHEUS_PASSWORD"] = "secret";
    const { logger } = await import("../src/logger.js");

    logger.info("authed");
    await new Promise((r) => setTimeout(r, 10));

    const headers = fetchMock.mock.calls[0][1].headers;
    const expected = "Basic " + Buffer.from("alice:secret").toString("base64");
    expect(headers["Authorization"]).toBe(expected);
  });

  it("merges fields into the JSON line", async () => {
    process.env["LOKI_PUSH_URL"] = "https://loki.example/loki/api/v1/push";
    process.env["LOKI_BATCH_SIZE"] = "1";
    const { logger } = await import("../src/logger.js");

    logger.info("Session summarized", { sessionId: "sess-1", chunks: 5 });
    await new Promise((r) => setTimeout(r, 10));

    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    const line = JSON.parse(body.streams[0].values[0][1]);
    expect(line.msg).toBe("Session summarized");
    expect(line.sessionId).toBe("sess-1");
    expect(line.chunks).toBe(5);
    expect(typeof line.ts).toBe("string");
  });

  it("survives a fetch failure without throwing", async () => {
    process.env["LOKI_PUSH_URL"] = "https://loki.example/loki/api/v1/push";
    process.env["LOKI_BATCH_SIZE"] = "1";
    fetchMock = vi.fn(async () => {
      throw new Error("network down");
    });
    globalThis.fetch = fetchMock as unknown as typeof globalThis.fetch;
    const { logger } = await import("../src/logger.js");

    expect(() => logger.info("offline")).not.toThrow();
    // Wait for the async push to settle
    await new Promise((r) => setTimeout(r, 20));
    expect(fetchMock).toHaveBeenCalledOnce();
  });
});
