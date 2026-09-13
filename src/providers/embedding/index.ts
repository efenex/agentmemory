import type { EmbeddingProvider } from "../../types.js";
import { detectEmbeddingProvider, getEnvVar } from "../../config.js";
import { GeminiEmbeddingProvider } from "./gemini.js";
import { OpenAIEmbeddingProvider } from "./openai.js";
import { VoyageEmbeddingProvider } from "./voyage.js";
import { CohereEmbeddingProvider } from "./cohere.js";
import { OpenRouterEmbeddingProvider } from "./openrouter.js";
import { LocalEmbeddingProvider } from "./local.js";
import { ClipEmbeddingProvider } from "./clip.js";
import {
  embeddingRequestsTotal,
  embeddingRequestDurationMs,
  embeddingBatchSize,
  metricsEnabled,
} from "../../telemetry/prometheus.js";
import { withSpan, tracingEnabled } from "../../telemetry/tracer.js";

export {
  GeminiEmbeddingProvider,
  OpenAIEmbeddingProvider,
  VoyageEmbeddingProvider,
  CohereEmbeddingProvider,
  OpenRouterEmbeddingProvider,
  LocalEmbeddingProvider,
  ClipEmbeddingProvider,
};

let imageEmbeddingProvider: EmbeddingProvider | null = null;

export function createImageEmbeddingProvider(): EmbeddingProvider | null {
  if (process.env["AGENTMEMORY_IMAGE_EMBEDDINGS"] !== "true") return null;
  if (imageEmbeddingProvider) return imageEmbeddingProvider;
  imageEmbeddingProvider = withMetrics(
    withDimensionGuard(new ClipEmbeddingProvider()),
  );
  return imageEmbeddingProvider;
}

export function createEmbeddingProvider(): EmbeddingProvider | null {
  const detected = detectEmbeddingProvider();
  if (!detected) return null;

  const wrap = (p: EmbeddingProvider) =>
    withMetrics(withDimensionGuard(p));

  switch (detected) {
    case "gemini":
      return wrap(new GeminiEmbeddingProvider(getEnvVar("GEMINI_API_KEY")!));
    case "openai":
      return wrap(new OpenAIEmbeddingProvider(getEnvVar("OPENAI_API_KEY")!));
    case "voyage":
      return wrap(new VoyageEmbeddingProvider(getEnvVar("VOYAGE_API_KEY")!));
    case "cohere":
      return wrap(new CohereEmbeddingProvider(getEnvVar("COHERE_API_KEY")!));
    case "openrouter":
      return wrap(new OpenRouterEmbeddingProvider(getEnvVar("OPENROUTER_API_KEY")!));
    case "local":
      return wrap(new LocalEmbeddingProvider());
    default:
      return null;
  }
}

// Wrong-dimension vectors corrupt the index silently: vector-index.ts
// returns 0 from cosineSimilarity on length mismatch instead of throwing,
// so a bad vector is stored, never matches anything, and the memory
// becomes invisible without an error. Catch it at the boundary.
export function withDimensionGuard(provider: EmbeddingProvider): EmbeddingProvider {
  const expected = provider.dimensions;
  const check = (v: Float32Array, where: string): Float32Array => {
    if (v.length !== expected) {
      throw new Error(
        `Embedding dimension mismatch in ${provider.name}.${where}: expected ${expected}, got ${v.length}`,
      );
    }
    return v;
  };
  // Preserve the provider's prototype chain so `instanceof` checks
  // against concrete classes (e.g. GeminiEmbeddingProvider) keep working.
  const wrapped = Object.create(provider) as EmbeddingProvider;
  wrapped.embed = async (t, tt) => check(await provider.embed(t, tt), "embed");
  wrapped.embedBatch = async (ts, tt) => {
    const out = await provider.embedBatch(ts, tt);
    out.forEach((v, i) => check(v, `embedBatch[${i}]`));
    return out;
  };
  if (provider.embedImage) {
    wrapped.embedImage = async (s: string) =>
      check(await provider.embedImage!(s), "embedImage");
  }
  return wrapped;
}

// Wrap an embedding provider with Prometheus instrumentation. Mirror
// of the withDimensionGuard pattern: preserves the prototype chain so
// `instanceof` checks against the concrete classes keep working.
// No-op overhead when metrics are disabled.
export function withMetrics(provider: EmbeddingProvider): EmbeddingProvider {
  if (!metricsEnabled() && !tracingEnabled()) return provider;
  const wrapped = Object.create(provider) as EmbeddingProvider;
  const label = provider.name;
  wrapped.embedBatch = async (texts, taskType) =>
    withSpan(
      "embedding.batch",
      {
        "gen_ai.system": label,
        "gen_ai.operation.name": "embed_batch",
        "agentmemory.batch_size": texts.length,
      },
      async () => {
        const t0 = Date.now();
        try {
          const out = await provider.embedBatch(texts, taskType);
          const dur = Date.now() - t0;
          if (metricsEnabled()) {
            embeddingRequestsTotal.inc({ provider: label, status: "success" });
            embeddingRequestDurationMs.observe({ provider: label }, dur);
            embeddingBatchSize.observe({ provider: label }, texts.length);
          }
          return out;
        } catch (err) {
          if (metricsEnabled()) {
            embeddingRequestsTotal.inc({ provider: label, status: "error" });
            embeddingRequestDurationMs.observe(
              { provider: label },
              Date.now() - t0,
            );
          }
          throw err;
        }
      },
    );
  // embed() calls embedBatch() in every concrete impl, so it's
  // already instrumented transitively. Leave it alone.
  if (provider.embedImage) {
    wrapped.embedImage = async (s: string) =>
      withSpan(
        "embedding.image",
        {
          "gen_ai.system": label,
          "gen_ai.operation.name": "embed_image",
        },
        async () => {
          const t0 = Date.now();
          try {
            const out = await provider.embedImage!(s);
            if (metricsEnabled()) {
              embeddingRequestsTotal.inc({
                provider: label,
                status: "success",
              });
              embeddingRequestDurationMs.observe(
                { provider: label },
                Date.now() - t0,
              );
            }
            return out;
          } catch (err) {
            if (metricsEnabled()) {
              embeddingRequestsTotal.inc({
                provider: label,
                status: "error",
              });
              embeddingRequestDurationMs.observe(
                { provider: label },
                Date.now() - t0,
              );
            }
            throw err;
          }
        },
      );
  }
  return wrapped;
}
