import { defineConfig } from "tsdown";

const hookEntries = [
  "src/hooks/session-start.ts",
  "src/hooks/prompt-submit.ts",
  "src/hooks/pre-tool-use.ts",
  "src/hooks/post-tool-use.ts",
  "src/hooks/post-tool-failure.ts",
  "src/hooks/pre-compact.ts",
  "src/hooks/subagent-start.ts",
  "src/hooks/subagent-stop.ts",
  "src/hooks/notification.ts",
  "src/hooks/task-completed.ts",
  "src/hooks/stop.ts",
  "src/hooks/session-end.ts",
  "src/hooks/post-commit.ts",
  "src/hooks/antigravity-bridge.ts",
];

const shared = {
  format: ["esm"] as const,
  target: "node20" as const,
  // Keep these as node_modules imports (deps.neverBundle). We never import
  // onnxruntime-{node,web} or sharp directly; they come in transitively
  // through @huggingface/transformers, which is lazy-loaded from
  // src/providers/embedding/{clip,local}.ts and src/state/reranker.ts.
  // Bundling inlines relative paths like
  // `../bin/napi-v3/darwin/arm64/onnxruntime_binding.node` that no longer
  // resolve from dist/. @huggingface/transformers is declared as an
  // optionalDependency in package.json so users can install it only when
  // they enable local embeddings / CLIP / reranker.
  deps: {
    neverBundle: [
      "@huggingface/transformers",
      // onnxruntime is @huggingface/transformers' native backend; bundling
      // it breaks its .node loading the same way it did under @xenova.
      "onnxruntime-node",
      "onnxruntime-web",
      "@anthropic-ai/claude-agent-sdk",
      "@anthropic-ai/sdk",
      // OTel SDK packages stay unbundled (loaded from node_modules at
      // runtime) so they don't perturb the bundle's chunk graph. Pulling
      // them in via tsdown introduces TDZ + __exportAll cycles between
      // cli.mjs, src-*.mjs and the synthetic chunk for version.ts.
      // Carried across upstream's `external:` -> `deps.neverBundle`
      // rename (0.9.29); the cycle bug returns if these are dropped.
      "@opentelemetry/sdk-trace-node",
      "@opentelemetry/sdk-trace-base",
      "@opentelemetry/exporter-trace-otlp-http",
      "@opentelemetry/resources",
      "@opentelemetry/semantic-conventions",
    ],
  },
  // Each entry is its own build, so the per-entry dts/deps timing notice
  // fires ~30 times and drowns the real output. It is informational only.
  inputOptions: {
    checks: { pluginTimings: false },
  },
};

export default defineConfig([
  {
    entry: ["src/index.ts"],
    outDir: "dist",
    ...shared,
    dts: true,
    clean: true,
    sourcemap: true,
    banner: { js: "#!/usr/bin/env node" },
  },
  {
    entry: ["src/cli.ts"],
    outDir: "dist",
    ...shared,
    clean: false,
    sourcemap: false,
  },
  {
    entry: ["src/mcp/standalone.ts"],
    outDir: "dist",
    ...shared,
    clean: false,
    sourcemap: false,
  },
  // One entry per config block prevents tsdown from hoisting shared
  // helpers into hashed chunks across hooks.
  ...hookEntries.map((entry) => ({
    entry: [entry],
    outDir: "dist/hooks",
    ...shared,
    clean: false,
    sourcemap: false,
  })),
  ...hookEntries.map((entry) => ({
    entry: [entry],
    outDir: "plugin/scripts",
    ...shared,
    clean: false,
    sourcemap: false,
  })),
]);
