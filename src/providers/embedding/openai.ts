import type { EmbeddingProvider, EmbeddingTaskType } from "../../types.js";
import { getEnvVar } from "../../config.js";
import { fetchWithTimeout } from "../_fetch.js";
import {
  DEFAULT_AZURE_API_VERSION,
  buildAuthHeaders,
  buildEmbeddingUrl,
  detectAzure,
  normalizeBaseUrl,
} from "../_openai-shared.js";
import { resolveDimensions } from "./_dimensions.js";

const DEFAULT_MODEL = "text-embedding-3-small";

/**
 * OpenAI-compatible embedding provider.
 *
 * Shares transport (URL builder, auth header, Azure detection) with
 * the OpenAI LLM provider via `_openai-shared` (#371). Same env knobs
 * pick up automatically: when `OPENAI_BASE_URL` points at an Azure
 * resource (`.openai.azure.com` hostname) the embedding request uses
 * Azure's `/embeddings` path with the `api-version` query param and
 * `api-key` header instead of `Authorization: Bearer`.
 *
 * Required env vars:
 *   OPENAI_API_KEY               — API key (fallback for OPENAI_EMBEDDING_API_KEY)
 *
 * Optional:
 *   OPENAI_BASE_URL              — base URL without path (default: https://api.openai.com).
 *                                  Azure: https://<resource>.openai.azure.com/openai/deployments/<deployment>
 *   OPENAI_EMBEDDING_BASE_URL    — embedding-specific base URL override (defaults
 *                                  to OPENAI_BASE_URL). Lets operators run
 *                                  embeddings on a separate endpoint from chat —
 *                                  e.g. local Ollama / LM Studio / llama.cpp /
 *                                  vLLM at http://localhost:1234 for unlimited
 *                                  free embeddings, while keeping chat
 *                                  completions on a rate-limited but high-quality
 *                                  hosted provider. Azure detection runs on
 *                                  whichever URL ends up selected.
 *   OPENAI_EMBEDDING_API_KEY     — separate API key for the embedding endpoint
 *                                  (defaults to OPENAI_API_KEY). Useful when the
 *                                  embedding endpoint requires a different key
 *                                  or no key at all (set to e.g. "local" for
 *                                  endpoints that ignore Authorization).
 *   OPENAI_API_VERSION           — Azure api-version query param (default: 2024-08-01-preview)
 *   OPENAI_EMBEDDING_MODEL       — model name (default: text-embedding-3-small)
 *   OPENAI_EMBEDDING_DIMENSIONS  — override reported dimensions (required for
 *                                  custom / self-hosted models not in the
 *                                  shared MODEL_DIMENSIONS table)
 */
export class OpenAIEmbeddingProvider implements EmbeddingProvider {
  readonly name = "openai";
  readonly dimensions: number;
  private apiKey: string;
  private baseUrl: string;
  private model: string;
  private isAzure: boolean;
  private azureApiVersion: string;
  // nomic-embed-text models are trained with task prefixes; embedding
  // without them lands queries and documents in mismatched subspaces
  // and measurably degrades recall. Detected from the model name.
  private usesTaskPrefixes: boolean;
  private explicitDimensions: boolean;
  private warnedTruncation = false;

  constructor(apiKey?: string) {
    // Separate API key path: caller-passed wins, then OPENAI_EMBEDDING_API_KEY,
    // then fall back to OPENAI_API_KEY. Allows e.g. a placeholder key for
    // local endpoints that ignore Authorization (most do).
    this.apiKey =
      apiKey ||
      getEnvVar("OPENAI_EMBEDDING_API_KEY") ||
      getEnvVar("OPENAI_API_KEY") ||
      "";
    if (!this.apiKey) {
      throw new Error(
        "API key is required (via constructor, OPENAI_EMBEDDING_API_KEY, or OPENAI_API_KEY)",
      );
    }
    // Embedding-specific base URL override; falls back to OPENAI_BASE_URL,
    // then normalizeBaseUrl's default. The chat-LLM path (src/providers/openai.ts)
    // still reads only OPENAI_BASE_URL, so setting OPENAI_EMBEDDING_BASE_URL
    // alone moves embeddings to the new endpoint without affecting chat.
    this.baseUrl = normalizeBaseUrl(
      getEnvVar("OPENAI_EMBEDDING_BASE_URL") || getEnvVar("OPENAI_BASE_URL"),
    );
    this.model = getEnvVar("OPENAI_EMBEDDING_MODEL") || DEFAULT_MODEL;
    this.dimensions = resolveDimensions(
      this.model,
      getEnvVar("OPENAI_EMBEDDING_DIMENSIONS"),
      "OPENAI_EMBEDDING_DIMENSIONS",
    );
    this.isAzure = detectAzure(this.baseUrl);
    this.azureApiVersion =
      getEnvVar("OPENAI_API_VERSION") || DEFAULT_AZURE_API_VERSION;
    this.usesTaskPrefixes = /nomic-embed/i.test(this.model);
    this.explicitDimensions = Boolean(
      getEnvVar("OPENAI_EMBEDDING_DIMENSIONS")?.trim(),
    );
  }

  async embed(
    text: string,
    taskType: EmbeddingTaskType = "document",
  ): Promise<Float32Array> {
    const [result] = await this.embedBatch([text], taskType);
    return result;
  }

  async embedBatch(
    texts: string[],
    taskType: EmbeddingTaskType = "document",
  ): Promise<Float32Array[]> {
    const url = buildEmbeddingUrl(
      this.baseUrl,
      this.isAzure,
      this.azureApiVersion,
    );
    // For models that support matryoshka representation learning (Qwen3
    // Embedding, OpenAI text-embedding-3, etc.) the OpenAI API accepts
    // a `dimensions` field to request a truncated vector instead of the
    // model's native size. Send it whenever OPENAI_EMBEDDING_DIMENSIONS
    // is set; the server is free to ignore it for non-matryoshka models
    // (their native dim is the only choice anyway).
    const input = this.usesTaskPrefixes
      ? texts.map(
          (t) =>
            `${taskType === "query" ? "search_query" : "search_document"}: ${t}`,
        )
      : texts;
    const body: Record<string, unknown> = {
      model: this.model,
      input,
    };
    if (this.explicitDimensions) {
      body.dimensions = this.dimensions;
    }
    const response = await fetchWithTimeout(url, {
      method: "POST",
      headers: buildAuthHeaders(this.apiKey, this.isAzure),
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`OpenAI embedding failed (${response.status}): ${err}`);
    }

    const data = (await response.json()) as {
      data: Array<{ embedding: number[] }>;
    };

    return data.data.map((d) => this.toConfiguredDims(d.embedding));
  }

  // Client-side matryoshka fallback: some servers (LM Studio) ignore the
  // `dimensions` request field and always return the model's native size.
  // For matryoshka-trained models the first N dims ARE the N-dim
  // embedding — truncate and L2-renormalize (cosine search downstream
  // assumes unit-ish vectors). Only fires when the caller explicitly set
  // OPENAI_EMBEDDING_DIMENSIONS below the returned size; a shorter-than-
  // expected vector still falls through to the dimension guard.
  private toConfiguredDims(embedding: number[]): Float32Array {
    if (!this.explicitDimensions || embedding.length <= this.dimensions) {
      return new Float32Array(embedding);
    }
    if (!this.warnedTruncation) {
      this.warnedTruncation = true;
      console.warn(
        `[embedding] ${this.model} returned ${embedding.length} dims; ` +
          `truncating to configured ${this.dimensions} (matryoshka) — ` +
          `server ignored the dimensions field`,
      );
    }
    const out = new Float32Array(embedding.slice(0, this.dimensions));
    let sumSq = 0;
    for (let i = 0; i < out.length; i++) sumSq += out[i] * out[i];
    const norm = Math.sqrt(sumSq);
    if (norm > 0) {
      for (let i = 0; i < out.length; i++) out[i] /= norm;
    }
    return out;
  }
}
