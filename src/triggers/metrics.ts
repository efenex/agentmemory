// Prometheus exporter HTTP server. Bound on a dedicated port instead of
// going through the iii engine trigger framework: the engine serializes
// every handler body as JSON, which would break the Prometheus text
// scrape format. A bare node:http server lets us emit the raw bytes with
// the correct text/plain; version=0.0.4 content type Prometheus expects.
//
// Port defaults to 9464 (the OTel /metrics convention; already published
// in docker/docker-compose.yml). Bound to AGENTMEMORY_METRICS_HOST,
// 127.0.0.1 by default — metrics are non-sensitive but should not leak
// to LAN by accident. When AGENTMEMORY_METRICS_ENABLED is false the
// server is never started.

import http from "node:http";
import { metricsEnabled, renderMetrics } from "../telemetry/prometheus.js";
import { logger } from "../logger.js";

let serverInstance: http.Server | null = null;

export function startMetricsServer(): http.Server | null {
  if (!metricsEnabled()) return null;
  if (serverInstance) return serverInstance;

  const host = process.env["AGENTMEMORY_METRICS_HOST"] || "0.0.0.0";
  const port = Number(process.env["AGENTMEMORY_METRICS_PORT"] || "9464");

  const server = http.createServer(async (req, res) => {
    if (req.method !== "GET") {
      res.writeHead(405, { "Content-Type": "text/plain" });
      res.end("method not allowed");
      return;
    }
    if (req.url === "/healthz") {
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end("ok");
      return;
    }
    if (req.url !== "/metrics") {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("not found");
      return;
    }
    try {
      const { contentType, body } = await renderMetrics();
      res.writeHead(200, { "Content-Type": contentType });
      res.end(body);
    } catch (err) {
      logger.error("metrics: render failed", {
        error: err instanceof Error ? err.message : String(err),
      });
      res.writeHead(500, { "Content-Type": "text/plain" });
      res.end("metrics render failed");
    }
  });

  server.on("error", (err) => {
    logger.error("metrics: server error", {
      message: err instanceof Error ? err.message : String(err),
      port,
    });
  });

  server.listen(port, host, () => {
    logger.info("metrics: exporter listening", { host, port });
  });

  // Don't keep the daemon alive just for this socket — primary servers
  // own the event loop. (Prometheus reconnects on each scrape anyway.)
  server.unref();

  serverInstance = server;
  return server;
}

export function stopMetricsServer(): Promise<void> {
  return new Promise((resolve) => {
    if (!serverInstance) {
      resolve();
      return;
    }
    serverInstance.close(() => {
      serverInstance = null;
      resolve();
    });
  });
}
