import type { CompressedObservation } from "../types.js";
import { stem } from "./stemmer.js";
import { getSynonyms } from "./synonyms.js";
import { segmentCjk, hasCjk } from "./cjk-segmenter.js";

interface IndexEntry {
  obsId: string;
  sessionId: string;
  termCount: number;
}

export class SearchIndex {
  private entries: Map<string, IndexEntry> = new Map();
  private invertedIndex: Map<string, Set<string>> = new Map();
  private docTermCounts: Map<string, Map<string, number>> = new Map();
  private totalDocLength = 0;
  private sortedTerms: string[] | null = null;

  private readonly k1 = 1.2;
  private readonly b = 0.75;

  add(obs: CompressedObservation): void {
    const terms = this.extractTerms(obs);
    const termFreq = new Map<string, number>();
    let termCount = 0;

    for (const term of terms) {
      termFreq.set(term, (termFreq.get(term) || 0) + 1);
      termCount++;
    }

    this.entries.set(obs.id, {
      obsId: obs.id,
      sessionId: obs.sessionId,
      termCount,
    });
    this.docTermCounts.set(obs.id, termFreq);
    this.totalDocLength += termCount;

    for (const term of termFreq.keys()) {
      if (!this.invertedIndex.has(term)) {
        this.invertedIndex.set(term, new Set());
      }
      this.invertedIndex.get(term)!.add(obs.id);
    }

    this.sortedTerms = null;
  }

  has(id: string): boolean {
    return this.entries.has(id);
  }

  remove(id: string): void {
    const entry = this.entries.get(id);
    if (!entry) return;

    const termFreq = this.docTermCounts.get(id);
    if (termFreq) {
      for (const term of termFreq.keys()) {
        const postingList = this.invertedIndex.get(term);
        if (postingList) {
          postingList.delete(id);
          if (postingList.size === 0) {
            this.invertedIndex.delete(term);
          }
        }
      }
      this.docTermCounts.delete(id);
    }

    this.totalDocLength = Math.max(0, this.totalDocLength - entry.termCount);
    this.entries.delete(id);
    this.sortedTerms = null;
  }

  search(
    query: string,
    limit = 20,
  ): Array<{ obsId: string; sessionId: string; score: number }> {
    const rawTerms = this.tokenize(query.toLowerCase());
    if (rawTerms.length === 0) return [];

    const N = this.entries.size;
    if (N === 0) return [];
    const avgDocLen = this.totalDocLength / N;

    const queryTerms: Array<{ term: string; weight: number }> = [];
    const seen = new Set<string>();
    for (const term of rawTerms) {
      if (!seen.has(term)) {
        seen.add(term);
        queryTerms.push({ term, weight: 1.0 });
      }
      for (const syn of getSynonyms(term)) {
        if (!seen.has(syn)) {
          seen.add(syn);
          queryTerms.push({ term: syn, weight: 0.7 });
        }
      }
    }

    const scores = new Map<string, number>();
    const sorted = this.getSortedTerms();

    for (const { term, weight } of queryTerms) {
      const matchingDocs = this.invertedIndex.get(term);
      if (matchingDocs) {
        const df = matchingDocs.size;
        const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);

        for (const obsId of matchingDocs) {
          const entry = this.entries.get(obsId)!;
          const docTerms = this.docTermCounts.get(obsId);
          const tf = docTerms?.get(term) || 0;
          const docLen = entry.termCount;

          const numerator = tf * (this.k1 + 1);
          const denominator =
            tf + this.k1 * (1 - this.b + this.b * (docLen / avgDocLen));
          const bm25Score = idf * (numerator / denominator) * weight;

          scores.set(obsId, (scores.get(obsId) || 0) + bm25Score);
        }
      }

      const startIdx = this.lowerBound(sorted, term);
      for (let si = startIdx; si < sorted.length; si++) {
        const indexTerm = sorted[si];
        if (!indexTerm.startsWith(term)) break;
        if (indexTerm === term) continue;

        const obsIds = this.invertedIndex.get(indexTerm)!;
        const prefixDf = obsIds.size;
        const prefixIdf =
          Math.log((N - prefixDf + 0.5) / (prefixDf + 0.5) + 1) * 0.5;
        for (const obsId of obsIds) {
          const entry = this.entries.get(obsId)!;
          const docTerms = this.docTermCounts.get(obsId);
          const tf = docTerms?.get(indexTerm) || 0;
          const docLen = entry.termCount;
          const numerator = tf * (this.k1 + 1);
          const denominator =
            tf + this.k1 * (1 - this.b + this.b * (docLen / avgDocLen));
          scores.set(
            obsId,
            (scores.get(obsId) || 0) + prefixIdf * (numerator / denominator) * weight,
          );
        }
      }
    }

    return Array.from(scores.entries())
      .map(([obsId, score]) => {
        const entry = this.entries.get(obsId)!;
        return { obsId, sessionId: entry.sessionId, score };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  get size(): number {
    return this.entries.size;
  }

  clear(): void {
    this.entries.clear();
    this.invertedIndex.clear();
    this.docTermCounts.clear();
    this.totalDocLength = 0;
    this.sortedTerms = null;
  }

  restoreFrom(other: SearchIndex): void {
    this.entries = new Map(
      Array.from(other.entries.entries()).map(([k, v]) => [k, { ...v }]),
    );
    this.invertedIndex = new Map(
      Array.from(other.invertedIndex.entries()).map(([k, v]) => [
        k,
        new Set(v),
      ]),
    );
    this.docTermCounts = new Map(
      Array.from(other.docTermCounts.entries()).map(([k, v]) => [
        k,
        new Map(v),
      ]),
    );
    this.totalDocLength = other.totalDocLength;
    this.sortedTerms = null;
  }

  serialize(): string {
    const entries = Array.from(this.entries.entries());
    const inverted = Array.from(this.invertedIndex.entries()).map(
      ([term, ids]) => [term, Array.from(ids)] as [string, string[]],
    );
    const docTerms = Array.from(this.docTermCounts.entries()).map(
      ([id, counts]) =>
        [id, Array.from(counts.entries())] as [string, [string, number][]],
    );
    return JSON.stringify({
      v: 2,
      entries,
      inverted,
      docTerms,
      totalDocLength: this.totalDocLength,
    });
  }

  static deserialize(json: string): SearchIndex {
    try {
      const idx = new SearchIndex();
      const data = JSON.parse(json);
      if (!data?.entries || !data?.inverted || !data?.docTerms) return idx;
      for (const [key, val] of data.entries) {
        idx.entries.set(key, val);
      }
      for (const [term, ids] of data.inverted) {
        idx.invertedIndex.set(term, new Set(ids));
      }
      for (const [id, counts] of data.docTerms) {
        idx.docTermCounts.set(id, new Map(counts));
      }
      const rawLen = Number(data.totalDocLength);
      idx.totalDocLength =
        Number.isFinite(rawLen) && rawLen >= 0 ? Math.floor(rawLen) : 0;
      return idx;
    } catch {
      return new SearchIndex();
    }
  }

  // Chunked persistence for large corpora. The legacy serialize() above
  // builds a single JSON string holding the whole index; at ~500k docs
  // that string exceeds V8's max-string-length (~1 GB on x64) and save
  // fails just like the vector index did before chunks-v1. The save path
  // now emits three independent chunk streams (entries / inverted /
  // docTerms) plus a meta record written last as the commit point. Each
  // chunk targets `targetBytes` of JSON, building incrementally with a
  // string-array + join to stay O(n) rather than O(n^2).
  // See [[project-2026-05-22-vector-persistence-followups]] item #4.

  get totalDocLengthForPersist(): number {
    return this.totalDocLength;
  }

  setTotalDocLengthFromLoad(n: number): void {
    this.totalDocLength = Number.isFinite(n) && n >= 0 ? Math.floor(n) : 0;
  }

  *serializeEntriesChunks(
    targetBytes = 16 * 1024 * 1024,
  ): IterableIterator<{ index: number; json: string }> {
    let buf: string[] = [];
    let bufLen = 2; // for the surrounding []
    let index = 0;
    for (const [key, val] of this.entries) {
      const itemJson = JSON.stringify([key, val]);
      if (buf.length > 0 && bufLen + itemJson.length + 1 > targetBytes) {
        yield { index, json: "[" + buf.join(",") + "]" };
        index++;
        buf = [];
        bufLen = 2;
      }
      buf.push(itemJson);
      bufLen += itemJson.length + 1;
    }
    if (buf.length > 0) yield { index, json: "[" + buf.join(",") + "]" };
  }

  *serializeInvertedChunks(
    targetBytes = 16 * 1024 * 1024,
  ): IterableIterator<{ index: number; json: string }> {
    // One inverted entry is `[term, [obsId1, obsId2, ...]]`. Hot terms
    // can have hundreds of thousands of postings; per-item JSON can be
    // tens of MB on its own. The byte-size guard below catches that —
    // we never pack multiple hot terms into one chunk.
    let buf: string[] = [];
    let bufLen = 2;
    let index = 0;
    for (const [term, ids] of this.invertedIndex) {
      const itemJson = JSON.stringify([term, Array.from(ids)]);
      if (buf.length > 0 && bufLen + itemJson.length + 1 > targetBytes) {
        yield { index, json: "[" + buf.join(",") + "]" };
        index++;
        buf = [];
        bufLen = 2;
      }
      buf.push(itemJson);
      bufLen += itemJson.length + 1;
    }
    if (buf.length > 0) yield { index, json: "[" + buf.join(",") + "]" };
  }

  *serializeDocTermsChunks(
    targetBytes = 16 * 1024 * 1024,
  ): IterableIterator<{ index: number; json: string }> {
    let buf: string[] = [];
    let bufLen = 2;
    let index = 0;
    for (const [id, counts] of this.docTermCounts) {
      const itemJson = JSON.stringify([id, Array.from(counts.entries())]);
      if (buf.length > 0 && bufLen + itemJson.length + 1 > targetBytes) {
        yield { index, json: "[" + buf.join(",") + "]" };
        index++;
        buf = [];
        bufLen = 2;
      }
      buf.push(itemJson);
      bufLen += itemJson.length + 1;
    }
    if (buf.length > 0) yield { index, json: "[" + buf.join(",") + "]" };
  }

  appendEntriesChunkJson(json: string): void {
    let data: unknown;
    try {
      data = JSON.parse(json);
    } catch {
      return;
    }
    if (!Array.isArray(data)) return;
    for (const row of data) {
      if (!Array.isArray(row) || row.length < 2) continue;
      const [key, val] = row;
      if (typeof key !== "string" || !val || typeof val !== "object") continue;
      this.entries.set(key, val as IndexEntry);
    }
    this.sortedTerms = null;
  }

  appendInvertedChunkJson(json: string): void {
    let data: unknown;
    try {
      data = JSON.parse(json);
    } catch {
      return;
    }
    if (!Array.isArray(data)) return;
    for (const row of data) {
      if (!Array.isArray(row) || row.length < 2) continue;
      const [term, ids] = row;
      if (typeof term !== "string" || !Array.isArray(ids)) continue;
      const existing = this.invertedIndex.get(term);
      if (existing) {
        for (const id of ids) existing.add(id);
      } else {
        this.invertedIndex.set(term, new Set(ids));
      }
    }
    this.sortedTerms = null;
  }

  appendDocTermsChunkJson(json: string): void {
    let data: unknown;
    try {
      data = JSON.parse(json);
    } catch {
      return;
    }
    if (!Array.isArray(data)) return;
    for (const row of data) {
      if (!Array.isArray(row) || row.length < 2) continue;
      const [id, counts] = row;
      if (typeof id !== "string" || !Array.isArray(counts)) continue;
      this.docTermCounts.set(id, new Map(counts));
    }
  }

  private extractTerms(obs: CompressedObservation): string[] {
    const parts = [
      obs.title,
      obs.subtitle || "",
      obs.narrative,
      ...obs.facts,
      ...obs.concepts,
      ...obs.files,
      obs.type,
    ];
    return this.tokenize(parts.join(" ").toLowerCase());
  }

  private tokenize(text: string): string[] {
    const cleaned = text.replace(/[^\p{L}\p{N}\s/.\\-_]/gu, " ");
    const out: string[] = [];
    for (const raw of cleaned.split(/\s+/)) {
      if (raw.length < 2) continue;
      if (hasCjk(raw)) {
        for (const seg of segmentCjk(raw)) {
          if (seg.length >= 1) out.push(seg);
        }
      } else {
        out.push(stem(raw));
      }
    }
    return out;
  }

  private getSortedTerms(): string[] {
    if (!this.sortedTerms) {
      this.sortedTerms = Array.from(this.invertedIndex.keys()).sort();
    }
    return this.sortedTerms;
  }

  private lowerBound(arr: string[], target: string): number {
    let lo = 0;
    let hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (arr[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
}
