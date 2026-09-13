export const REFLECT_SYSTEM = `You are a higher-order reasoning engine. Given a cluster of related concepts, facts, lessons, and action outcomes, synthesize cross-cutting insights that span multiple individual memories.

Output format (XML):
<insights>
  <insight confidence="0.0-1.0" title="Short descriptive title">
    The higher-order observation or principle. Should be actionable and non-obvious — something that only becomes visible when viewing multiple memories together.
  </insight>
</insights>

Rules:
- Identify patterns, principles, or strategies that span 2+ source items
- Confidence reflects how well-supported the insight is across sources
- Title should be a concise label (under 60 chars)
- Content should be the actual observation (1-3 sentences)
- Prefer actionable insights over abstract summaries
- Skip insights that merely restate a single source item
- Always emit confidence attribute before title attribute`;

// Default prompt budget (~3K tokens at ~4 chars/token). Sized to fit a
// modest local context window (8K+) alongside the 4K max_tokens output,
// and to keep per-call latency sane on laptop inference. Override with
// AGENTMEMORY_REFLECT_PROMPT_CHARS when the model is loaded with a larger
// context (e.g. LM Studio qwen3-vl at 16K/32K).
const DEFAULT_REFLECT_PROMPT_CHARS = 12000;
// No single fact/lesson/crystal may consume more than this, so one giant
// item can't starve the rest of the cluster out of the budget.
const MAX_ITEM_CHARS = 800;

function truncateItem(s: string): string {
  return s.length > MAX_ITEM_CHARS ? `${s.slice(0, MAX_ITEM_CHARS)}…` : s;
}

export function buildReflectPrompt(cluster: {
  concepts: string[];
  facts: Array<{ fact: string; confidence: number }>;
  lessons: Array<{ content: string; confidence: number }>;
  crystalNarratives: string[];
}): string {
  // #814 follow-up: a high-degree concept cluster can match hundreds of
  // lessons, producing a 20K+ token prompt. Local runtimes (LM Studio,
  // vLLM) reject an over-length prompt with a fast HTTP 400 rather than
  // truncating — and 3 such 400s trip the provider circuit breaker,
  // fast-failing every remaining cluster. Bound the prompt to a char
  // budget, spending it highest-confidence-first so the most load-bearing
  // evidence survives the cap.
  const maxChars = Math.max(
    2000,
    parseInt(process.env["AGENTMEMORY_REFLECT_PROMPT_CHARS"] || "", 10) ||
      DEFAULT_REFLECT_PROMPT_CHARS,
  );

  const header = `## Concept Cluster: ${cluster.concepts.join(", ")}`;
  const sections: string[] = [header];
  let budget = maxChars - header.length;

  const addBounded = (
    heading: string,
    lines: string[],
  ): void => {
    const kept: string[] = [];
    for (const line of lines) {
      if (line.length + 1 > budget) break;
      kept.push(line);
      budget -= line.length + 1;
    }
    if (kept.length > 0) sections.push(heading, ...kept);
  };

  const factsByConf = [...cluster.facts].sort(
    (a, b) => b.confidence - a.confidence,
  );
  addBounded(
    "\n## Known Facts",
    factsByConf.map(
      (f) => `- [confidence=${f.confidence}] ${truncateItem(f.fact)}`,
    ),
  );

  const lessonsByConf = [...cluster.lessons].sort(
    (a, b) => b.confidence - a.confidence,
  );
  addBounded(
    "\n## Lessons Learned",
    lessonsByConf.map(
      (l) => `- [confidence=${l.confidence}] ${truncateItem(l.content)}`,
    ),
  );

  addBounded(
    "\n## Completed Work Summaries",
    cluster.crystalNarratives.map((n) => `- ${truncateItem(n)}`),
  );

  return `Synthesize higher-order insights from this cluster of related memories:\n\n${sections.join("\n")}`;
}
