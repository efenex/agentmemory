/**
 * Embedding bake-off: rank candidate embedding models on retrieval quality
 * before committing to a full-corpus backfill.
 *
 * Why a dedicated harness rather than benchmark/real-embeddings-eval.ts:
 * that one is hard-wired to LocalEmbeddingProvider and one synthetic
 * dataset. Choosing a model to re-embed ~130k real observations with needs
 * (a) several candidates in one run under identical conditions, and (b) a
 * corpus that looks like the real one. So this runs every candidate over
 * BOTH labelled corpora we have:
 *
 *   synthetic   — benchmark/dataset.ts, 240 obs / 20 queries, gold at
 *                 OBSERVATION level. Categories (exact/semantic/entity/
 *                 cross-session) make it good at showing WHERE a model wins.
 *   coding-life — eval/data/coding-agent-life-v1, 15 sessions / 15
 *                 hand-graded queries, gold at SESSION level. Small, but
 *                 it is real coding-agent transcript text, which is what
 *                 this install actually stores.
 *
 * Read them together. 35 labelled queries cannot separate two models a
 * couple of points apart -- treat the output as "is this model in the
 * right class", not as a precise ranking, and prefer the model that wins
 * on BOTH corpora.
 *
 * Usage:
 *   OLLAMA_URL=http://127.0.0.1:11434/v1 npx tsx benchmark/embedding-bakeoff.ts
 *   ... --candidates nomic-embed-text,qwen3-embedding:0.6b
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { SearchIndex } from "../src/state/search-index.js";
import { VectorIndex } from "../src/state/vector-index.js";
import { HybridSearch } from "../src/state/hybrid-search.js";
import { OpenAIEmbeddingProvider } from "../src/providers/embedding/openai.js";
import type { CompressedObservation, EmbeddingProvider } from "../src/types.js";
import { generateDataset } from "./dataset.js";

const ROOT = join(import.meta.dirname, "..");
const OLLAMA_URL = process.env.OLLAMA_URL ?? "http://127.0.0.1:11434/v1";

// Candidate models. `dims` is what we ASK for; models that support
// matryoshka honour it, the rest return native size and the provider
// truncates client-side (and warns). Keep the dims we would actually
// deploy with -- a model measured at 1024 tells you nothing about how it
// performs truncated to 256.
type Candidate = { label: string; model: string; dims: number };
const ALL_CANDIDATES: Candidate[] = [
  { label: "nomic-embed-text (768d)", model: "nomic-embed-text:latest", dims: 768 },
  { label: "embeddinggemma (768d)", model: "embeddinggemma:latest", dims: 768 },
  { label: "qwen3-embedding:0.6b (1024d)", model: "qwen3-embedding:0.6b", dims: 1024 },
  { label: "qwen3-embedding:8b (1024d)", model: "qwen3-embedding:8b", dims: 1024 },
];

// ---------- metrics ----------
const avg = (n: number[]) => (n.length ? n.reduce((a, b) => a + b, 0) / n.length : 0);
const pct = (n: number) => (n * 100).toFixed(1) + "%";

function recall(retrieved: string[], relevant: Set<string>, k: number): number {
  if (relevant.size === 0) return 1;
  const top = new Set(retrieved.slice(0, k));
  let hits = 0;
  for (const id of relevant) if (top.has(id)) hits++;
  return hits / relevant.size;
}
function mrr(retrieved: string[], relevant: Set<string>): number {
  for (let i = 0; i < retrieved.length; i++) if (relevant.has(retrieved[i])) return 1 / (i + 1);
  return 0;
}
function dcg(rel: boolean[], k: number): number {
  let s = 0;
  for (let i = 0; i < Math.min(k, rel.length); i++) s += (rel[i] ? 1 : 0) / Math.log2(i + 2);
  return s;
}
function ndcg(retrieved: string[], relevant: Set<string>, k: number): number {
  const ideal = dcg(Array.from({ length: Math.min(k, relevant.size) }, () => true), k);
  return ideal === 0 ? 0 : dcg(retrieved.slice(0, k).map((id) => relevant.has(id)), k) / ideal;
}

function mockKV() {
  const store = new Map<string, Map<string, unknown>>();
  return {
    get: async <T>(s: string, k: string): Promise<T | null> =>
      (store.get(s)?.get(k) as T) ?? null,
    set: async <T>(s: string, k: string, d: T): Promise<T> => {
      if (!store.has(s)) store.set(s, new Map());
      store.get(s)!.set(k, d);
      return d;
    },
    delete: async (s: string, k: string): Promise<void> => {
      store.get(s)?.delete(k);
    },
    list: async <T>(s: string): Promise<T[]> =>
      store.has(s) ? (Array.from(store.get(s)!.values()) as T[]) : [],
  };
}

// ---------- corpora ----------
type Corpus = {
  name: string;
  observations: CompressedObservation[];
  queries: { query: string; category: string; relevant: Set<string> }[];
};

function syntheticCorpus(): Corpus {
  const { observations, queries } = generateDataset();
  return {
    name: "synthetic",
    observations,
    queries: queries.map((q) => ({
      query: q.query,
      category: q.category,
      relevant: new Set(q.relevantObsIds),
    })),
  };
}

/**
 * coding-agent-life-v1 grades at SESSION level, so each session becomes one
 * observation and the gold set is the gold session ids. That makes recall@k
 * here "did the right session surface", which is the question that matters
 * for a memory store anyway.
 */
function codingLifeCorpus(): Corpus {
  const sessions = JSON.parse(
    readFileSync(join(ROOT, "eval/data/coding-agent-life-v1/sessions.json"), "utf-8"),
  ) as { id: string; timestamp: string; content: string }[];
  const queries = JSON.parse(
    readFileSync(join(ROOT, "eval/data/coding-agent-life-v1/queries.json"), "utf-8"),
  ) as { id: string; type: string; question: string; goldSessionIds: string[] }[];

  // SearchIndex.extractTerms spreads facts/concepts/files unconditionally,
  // so every one must be a real array even when empty.
  const observations: CompressedObservation[] = sessions.map((s) => ({
    id: s.id,
    sessionId: s.id,
    type: "session",
    title: s.content.slice(0, 80),
    subtitle: "",
    narrative: s.content,
    facts: [],
    concepts: [],
    files: [],
    importance: 5,
    timestamp: s.timestamp,
  }) as unknown as CompressedObservation);

  return {
    name: "coding-life",
    observations,
    queries: queries.map((q) => ({
      query: q.question,
      category: q.type,
      relevant: new Set(q.goldSessionIds),
    })),
  };
}

const obsText = (o: CompressedObservation) =>
  [o.title, (o as { subtitle?: string }).subtitle ?? "", o.narrative, ...(o.facts ?? []), ...(o.concepts ?? [])].join(" ");

// ---------- evaluation ----------
type Row = {
  system: string;
  corpus: string;
  recall5: number;
  recall10: number;
  ndcg10: number;
  mrr: number;
  embedMs: number;
  searchMs: number;
};

async function evaluate(
  system: string,
  corpus: Corpus,
  provider: EmbeddingProvider | null,
): Promise<Row> {
  const kv = mockKV();
  const bm25 = new SearchIndex();
  const vector = provider ? new VectorIndex() : null;

  for (const o of corpus.observations) {
    bm25.add(o);
    await kv.set(`mem:obs:${o.sessionId}`, o.id, o);
  }

  const tEmbed = performance.now();
  if (provider && vector) {
    const B = 32;
    for (let i = 0; i < corpus.observations.length; i += B) {
      const batch = corpus.observations.slice(i, i + B);
      const vecs = await provider.embedBatch(batch.map(obsText));
      batch.forEach((o, j) => vector.add(o.id, o.sessionId, vecs[j]));
    }
  }
  const embedMs = performance.now() - tEmbed;

  // Weights left at the shipped defaults on purpose: this measures the
  // MODEL, not a per-model weight tune. Tuning weights per candidate would
  // make the comparison unfalsifiable.
  const hybrid = new HybridSearch(bm25, vector, provider, kv as never);

  const r5: number[] = [], r10: number[] = [], nd: number[] = [], mr: number[] = [];
  const tSearch = performance.now();
  for (const q of corpus.queries) {
    const hits = await hybrid.search(q.query, 20);
    const ids = hits.map((h) => h.observation.id);
    r5.push(recall(ids, q.relevant, 5));
    r10.push(recall(ids, q.relevant, 10));
    nd.push(ndcg(ids, q.relevant, 10));
    mr.push(mrr(ids, q.relevant));
  }
  const searchMs = (performance.now() - tSearch) / corpus.queries.length;

  return {
    system, corpus: corpus.name,
    recall5: avg(r5), recall10: avg(r10), ndcg10: avg(nd), mrr: avg(mr),
    embedMs, searchMs,
  };
}

function makeProvider(c: Candidate): EmbeddingProvider {
  // The provider reads its config from env at construction time.
  process.env.OPENAI_EMBEDDING_BASE_URL = OLLAMA_URL;
  process.env.OPENAI_EMBEDDING_MODEL = c.model;
  process.env.OPENAI_EMBEDDING_DIMENSIONS = String(c.dims);
  // ollama ignores the key but the provider requires one to be present.
  process.env.OPENAI_EMBEDDING_API_KEY = "ollama";
  return new OpenAIEmbeddingProvider();
}

async function main(): Promise<void> {
  const argIdx = process.argv.indexOf("--candidates");
  const wanted = argIdx > -1 ? process.argv[argIdx + 1].split(",") : null;
  const candidates = wanted
    ? ALL_CANDIDATES.filter((c) => wanted.some((w) => c.model.startsWith(w)))
    : ALL_CANDIDATES;

  const corpora = [syntheticCorpus(), codingLifeCorpus()];
  const rows: Row[] = [];

  for (const corpus of corpora) {
    console.error(`\n### corpus ${corpus.name}: ${corpus.observations.length} obs / ${corpus.queries.length} queries`);
    // BM25-only baseline == exactly what this install runs today
    // (EMBEDDING_PROVIDER=none). Every candidate is judged against it.
    console.error("  [baseline] bm25-only");
    rows.push(await evaluate("bm25-only (today)", corpus, null));
    for (const c of candidates) {
      console.error(`  [candidate] ${c.label}`);
      try {
        rows.push(await evaluate(c.label, corpus, makeProvider(c)));
      } catch (err) {
        console.error(`    FAILED: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }

  const header = "| system | corpus | recall@5 | recall@10 | NDCG@10 | MRR | embed ms | search ms/q |";
  const lines = [header, "|---|---|---|---|---|---|---|---|"];
  for (const r of rows) {
    lines.push(
      `| ${r.system} | ${r.corpus} | ${pct(r.recall5)} | ${pct(r.recall10)} | ${pct(r.ndcg10)} | ${pct(r.mrr)} | ${r.embedMs.toFixed(0)} | ${r.searchMs.toFixed(1)} |`,
    );
  }
  const out = lines.join("\n");
  console.log("\n" + out);
  writeFileSync(join(ROOT, "benchmark/results/embedding-bakeoff.md"), out + "\n");
  writeFileSync(join(ROOT, "benchmark/results/embedding-bakeoff.json"), JSON.stringify(rows, null, 2));
  console.error("\nwrote benchmark/results/embedding-bakeoff.{md,json}");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
