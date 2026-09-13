// Int8 scalar quantization helpers. Stored vectors are int8 arrays plus
// a per-vector float scale; reconstructed value f[i] ≈ q[i] * scale.
//
// We adopted int8 quantization on 2026-05-23 to cut the in-memory
// footprint of the vector index by 4× (float32 → int8). For our
// ~490k-vector / 1024-dim corpus this drops RAM use from ~2 GB pure
// vector data to ~500 MB, which in turn keeps the chunked vector save
// under iii-engine's memory pressure (previously OOM-killed at ~chunk
// 288 of 325 during a full save). Recall on top-K stays exact for our
// use case (relative ordering preserved up to quantization noise on
// unit-norm vectors — see [[project-2026-05-22-vector-persistence-followups]]
// item #3 for the empirical alternatives we ruled out).

// Normalizes the input to unit length and quantizes the result to int8.
// Returning the scale is what lets a later int8 dot product reconstruct
// the cosine similarity exactly: cos(a,b) = scale_a * scale_b * q_a·q_b
// because we forced ‖a‖ = ‖b‖ = 1 before quantizing.
//
// Normalizing here (rather than at the call site) is what gives us the
// same semantics as the legacy cosineSimilarity, which divided dot by
// ‖a‖·‖b‖ at search time. For unit-norm inputs (e.g. OpenAI embeddings,
// already normalized) the divide is a no-op so we don't lose anything.
function quantizeNormalized(arr: Float32Array): { q: Int8Array; scale: number } {
  let l2sq = 0;
  for (let i = 0; i < arr.length; i++) l2sq += arr[i] * arr[i];
  const l2 = Math.sqrt(l2sq);
  const q = new Int8Array(arr.length);
  if (l2 === 0) return { q, scale: 0 };
  // After normalization every coordinate is in [-1, 1], so the largest
  // absolute coordinate sets the int8 quantization scale.
  let maxAbs = 0;
  for (let i = 0; i < arr.length; i++) {
    const a = Math.abs(arr[i]) / l2;
    if (a > maxAbs) maxAbs = a;
  }
  if (maxAbs === 0) return { q, scale: 0 };
  const invScale = 127 / (l2 * maxAbs);
  for (let i = 0; i < arr.length; i++) {
    let v = Math.round(arr[i] * invScale);
    if (v > 127) v = 127;
    else if (v < -128) v = -128;
    q[i] = v;
  }
  return { q, scale: maxAbs / 127 };
}

function int8ToBase64(arr: Int8Array): string {
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString(
    "base64",
  );
}

function base64ToInt8(b64: string): Int8Array {
  const buf = Buffer.from(b64, "base64");
  // Copy out of pooled Buffer so the Int8Array owns its memory.
  const copy = new ArrayBuffer(buf.byteLength);
  Buffer.from(copy).set(buf);
  return new Int8Array(copy);
}

// Float32 base64 — kept for back-compat with `chunks-v2` (pre-quant) on-disk
// snapshots. Loader detects the format from the record shape and dequantizes
// → re-quantizes into the new in-memory representation.
function base64ToFloat32(b64: string): Float32Array {
  const buf = Buffer.from(b64, "base64");
  const copy = new ArrayBuffer(buf.byteLength);
  Buffer.from(copy).set(buf);
  return new Float32Array(copy);
}

function int8DotProduct(a: Int8Array, b: Int8Array): number {
  // Each product is bounded by 127² = 16129, so for typical embedding
  // dimensions (≤ 16k) the accumulator stays well inside safe integer
  // range — no overflow risk even at dim 100k+.
  let acc = 0;
  const len = a.length;
  for (let i = 0; i < len; i++) acc += a[i] * b[i];
  return acc;
}

// What we actually want to score is the true cosine between the original
// float32 vectors. With int8 quantization (scale_a, q_a) and (scale_b, q_b),
// the dequantized inner product is `scale_a * scale_b * int8DotProduct(q_a, q_b)`.
// For unit-norm inputs (OpenAI embeddings are unit-norm) this is the cosine
// similarity directly; for non-unit inputs it's the raw inner product, which
// is what the legacy cosineSimilarity returned for non-unit cases too —
// preserves call-site semantics.

interface QuantizedEntry {
  quantized: Int8Array;
  scale: number;
  sessionId: string;
}

export class VectorIndex {
  private vectors: Map<string, QuantizedEntry> = new Map();

  add(obsId: string, sessionId: string, embedding: Float32Array): void {
    const { q, scale } = quantizeNormalized(embedding);
    this.vectors.set(obsId, { quantized: q, scale, sessionId });
  }

  remove(obsId: string): void {
    this.vectors.delete(obsId);
  }

  has(obsId: string): boolean {
    return this.vectors.has(obsId);
  }

  search(
    query: Float32Array,
    limit = 20,
  ): Array<{ obsId: string; sessionId: string; score: number }> {
    const { q: queryQ, scale: queryScale } = quantizeNormalized(query);

    const results: Array<{
      obsId: string;
      sessionId: string;
      score: number;
    }> = [];
    let minScore = -Infinity;

    for (const [obsId, entry] of this.vectors) {
      if (entry.quantized.length !== queryQ.length) continue;
      // Zero entries (or a zero query) score 0 against everything — match
      // the legacy cosineSimilarity contract which returned 0 for
      // degenerate norms rather than silently dropping the entry.
      const score =
        queryScale === 0 || entry.scale === 0
          ? 0
          : queryScale * entry.scale * int8DotProduct(queryQ, entry.quantized);
      if (results.length < limit) {
        results.push({ obsId, sessionId: entry.sessionId, score });
        if (results.length === limit) {
          results.sort((a, b) => a.score - b.score);
          minScore = results[0].score;
        }
      } else if (score > minScore) {
        results[0] = { obsId, sessionId: entry.sessionId, score };
        results.sort((a, b) => a.score - b.score);
        minScore = results[0].score;
      }
    }

    results.sort((a, b) => b.score - a.score);
    return results;
  }

  get size(): number {
    return this.vectors.size;
  }

  // Walks every stored vector and returns the obsIds whose dimension
  // doesn't match `expected`, plus the set of distinct dimensions seen.
  // Used by the persistence-restore guard in src/index.ts to refuse
  // loading any index containing wrong-dimension vectors — including
  // legacy on-disk indexes written before the live-API dimension guard
  // existed (where a mid-session provider swap could mix dimensions
  // inside a single index). Empty `mismatches` plus a single-entry
  // `seenDimensions` matching `expected` is the only clean state.
  validateDimensions(
    expected: number,
  ): { mismatches: Array<{ obsId: string; dim: number }>; seenDimensions: Set<number> } {
    const mismatches: Array<{ obsId: string; dim: number }> = [];
    const seenDimensions = new Set<number>();
    for (const [obsId, entry] of this.vectors) {
      const dim = entry.quantized.length;
      seenDimensions.add(dim);
      if (dim !== expected) {
        mismatches.push({ obsId, dim });
      }
    }
    return { mismatches, seenDimensions };
  }

  clear(): void {
    this.vectors.clear();
  }

  restoreFrom(other: VectorIndex): void {
    const src = (other as any).vectors as Map<string, QuantizedEntry>;
    this.vectors = new Map();
    for (const [obsId, entry] of src) {
      this.vectors.set(obsId, {
        quantized: new Int8Array(entry.quantized),
        scale: entry.scale,
        sessionId: entry.sessionId,
      });
    }
  }

  // Serialize the whole index as a single JSON string. Kept for the
  // legacy single-blob save path; new persistence uses serializeChunks
  // below so we don't trip V8's max-string-length on large corpora.
  serialize(): string {
    const data: Array<[string, { q: string; s: number; sid: string }]> = [];
    for (const [obsId, entry] of this.vectors) {
      data.push([
        obsId,
        {
          q: int8ToBase64(entry.quantized),
          s: entry.scale,
          sid: entry.sessionId,
        },
      ]);
    }
    return JSON.stringify(data);
  }

  static deserialize(json: string): VectorIndex {
    const idx = new VectorIndex();
    let data: unknown;
    try {
      data = JSON.parse(json);
    } catch {
      return idx;
    }
    if (!Array.isArray(data)) return idx;
    for (const row of data) {
      try {
        if (!Array.isArray(row) || row.length < 2) continue;
        const [obsId, entry] = row;
        if (typeof obsId !== "string" || !entry) continue;
        // New (int8) shape: { q, s, sid }
        if (
          typeof entry.q === "string" &&
          typeof entry.s === "number" &&
          typeof entry.sid === "string"
        ) {
          idx.vectors.set(obsId, {
            quantized: base64ToInt8(entry.q),
            scale: entry.s,
            sessionId: entry.sid,
          });
          continue;
        }
        // Legacy (float32) shape: { embedding, sessionId } — dequantize +
        // re-quantize so the in-memory representation is uniform.
        if (
          typeof entry.embedding === "string" &&
          typeof entry.sessionId === "string"
        ) {
          const f32 = base64ToFloat32(entry.embedding);
          const { q, scale } = quantizeNormalized(f32);
          idx.vectors.set(obsId, {
            quantized: q,
            scale,
            sessionId: entry.sessionId,
          });
        }
      } catch {
        continue;
      }
    }
    return idx;
  }

  // Append one chunk's JSON payload into this index. Used by the
  // chunked-load path in IndexPersistence — see [[project-2026-05-22-vector-index-hang]].
  // Handles both new (int8) and legacy (float32) record shapes so a
  // chunks-v2 on-disk snapshot can still be loaded on first boot after
  // the quantization upgrade.
  appendChunkJson(json: string): void {
    let data: unknown;
    try {
      data = JSON.parse(json);
    } catch {
      return;
    }
    if (!Array.isArray(data)) return;
    for (const row of data) {
      try {
        if (!Array.isArray(row) || row.length < 2) continue;
        const [obsId, entry] = row;
        if (typeof obsId !== "string" || !entry) continue;
        if (
          typeof entry.q === "string" &&
          typeof entry.s === "number" &&
          typeof entry.sid === "string"
        ) {
          this.vectors.set(obsId, {
            quantized: base64ToInt8(entry.q),
            scale: entry.s,
            sessionId: entry.sid,
          });
          continue;
        }
        if (
          typeof entry.embedding === "string" &&
          typeof entry.sessionId === "string"
        ) {
          const f32 = base64ToFloat32(entry.embedding);
          const { q, scale } = quantizeNormalized(f32);
          this.vectors.set(obsId, {
            quantized: q,
            scale,
            sessionId: entry.sessionId,
          });
        }
      } catch {
        continue;
      }
    }
  }

  // Yields one JSON chunk at a time so the caller can write+await
  // between chunks. Entries-per-chunk is sized from the first vector's
  // dimension to target ~`targetBytes` of JSON per chunk; all vectors in
  // a healthy index share the same dim (validateDimensions enforces this).
  *serializeChunks(
    targetBytes = 8 * 1024 * 1024,
  ): IterableIterator<{ index: number; json: string }> {
    let entriesPerChunk = 4000;
    const firstEntry = this.vectors.values().next().value;
    if (firstEntry) {
      const sampleRow: [string, { q: string; s: number; sid: string }] = [
        "00000000-0000-0000-0000-000000000000",
        {
          q: int8ToBase64(firstEntry.quantized),
          s: firstEntry.scale,
          sid: firstEntry.sessionId,
        },
      ];
      const perEntryBytes = JSON.stringify(sampleRow).length + 2;
      if (perEntryBytes > 0) {
        entriesPerChunk = Math.max(
          100,
          Math.floor(targetBytes / perEntryBytes),
        );
      }
    }

    let index = 0;
    let buf: Array<[string, { q: string; s: number; sid: string }]> = [];
    for (const [obsId, entry] of this.vectors) {
      buf.push([
        obsId,
        {
          q: int8ToBase64(entry.quantized),
          s: entry.scale,
          sid: entry.sessionId,
        },
      ]);
      if (buf.length >= entriesPerChunk) {
        yield { index, json: JSON.stringify(buf) };
        index++;
        buf = [];
      }
    }
    if (buf.length > 0) yield { index, json: JSON.stringify(buf) };
  }
}
