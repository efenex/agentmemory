#!/usr/bin/env bash
# Import repo docs declared in `.agentmemory-ingest` at each repo's root
# into agentmemory's memory store. Companion to import-claude-memory.sh:
# that one handles ~/.claude/projects/*/memory/ files (user-curated
# memories Claude wrote); this one handles in-repo design docs the team
# wrote (architecture.md, agent specs, runbooks, etc.).
#
# Manifest format (`.agentmemory-ingest` at repo root):
#   gitignore-style. One glob per line. `!` prefix excludes. `#` is a
#   comment. Excludes win over includes. Globs are relative to repo
#   root. `**` matches zero or more directory components.
#
#   Example:
#     docs/*.md
#     docs/agents/*.md
#     docs/plans/session-handoff-*.md     # session handoffs — see V2-B
#     !docs/TODO.md                       # living index, exclude
#     !docs/external-review-*.md          # dated snapshots, exclude
#
# V2 behavior:
#   - **Chunking** (V2-A): files larger than AGENTMEMORY_DOC_MAX_BYTES
#     (default 50000) are split by `##` then `###` then paragraph
#     boundaries, with greedy coalescing to ~70% of MAX_BYTES so we
#     don't drown the corpus in 1KB shards. Lossless excerpts only — no
#     LLM summarization. Each chunk POSTs as a separate Memory with a
#     stable `chunk N/M` and `group <sha8>:N` token in the content
#     header so agentmemory's Jaccard-supersede pairs chunk-N-old to
#     chunk-N-new across re-imports.
#   - **Handoff routing** (V2-B): files matching
#     `**/session-handoff-YYYY-MM-DD*.md` get content prefix
#     `[Session handoff]` and `session:<id>` concept tags linking them
#     to every session for that project on that date (matched via
#     /agentmemory/sessions). Falls back to V1 [Repo doc] routing when
#     no session matches.
#   - **Sidecar key** (V2-C): keyed by `<project-basename>:<rel-path>`
#     not absolute path, so repo moves don't trigger phantom
#     re-imports. One-time migration of old `/...` keys runs
#     automatically on first V2 invocation.
#
# Repo discovery:
#   Scans ~/.claude/projects/*/ slugs and reverses each one back to the
#   real on-disk repo path. Only repos with a `.agentmemory-ingest`
#   manifest at the root get scanned. Pass --repo <path> to force a
#   specific repo without going through slug resolution.
#
# Idempotency:
#   Per-file SHA-256 sidecar at ~/.agentmemory/.import-doc-hashes.json.
#   Files whose content hash matches the previous import are skipped
#   entirely (all chunks). Hash is recorded only when ALL chunks of the
#   file POST successfully — failures retry on the next run.
#
# Cost: $0 chat tokens. Embedding tokens scale with imported chunk
# content (one embed call per posted memory, batched on the server
# side per PR #504).
#
# Usage:
#   scripts/import-repo-docs.sh --dry-run
#   scripts/import-repo-docs.sh --changed-only
#   scripts/import-repo-docs.sh --repo /path/to/repo            # one repo only
#   scripts/import-repo-docs.sh --project-pattern gitops        # filter
#   scripts/import-repo-docs.sh                                 # force re-import

set -euo pipefail

URL="${AGENTMEMORY_URL:-http://localhost:3111}"
PROJECTS_DIR="${CLAUDE_PROJECTS_DIR:-$HOME/.claude/projects}"
HASH_FILE="${AGENTMEMORY_DOC_HASHES:-$HOME/.agentmemory/.import-doc-hashes.json}"
MAX_BYTES="${AGENTMEMORY_DOC_MAX_BYTES:-50000}"
# Below this fraction-of-MAX_BYTES, the coalescer keeps packing adjacent
# sections rather than cutting. 0.7 = aim for chunks in [35KB, 50KB].
COALESCE_TARGET="${AGENTMEMORY_DOC_COALESCE_TARGET:-0.7}"
DRY_RUN=0
CHANGED_ONLY=0
PROJECT_PATTERN=""
EXPLICIT_REPO=""
LIMIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)          DRY_RUN=1; shift ;;
    --changed-only)     CHANGED_ONLY=1; shift ;;
    --project-pattern)  PROJECT_PATTERN="${2:?--project-pattern needs a regex}"; shift 2 ;;
    --repo)             EXPLICIT_REPO="${2:?--repo needs a path}"; shift 2 ;;
    --limit)            LIMIT="${2:?--limit needs a number}"; shift 2 ;;
    --max-bytes)        MAX_BYTES="${2:?--max-bytes needs a number}"; shift 2 ;;
    -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

for bin in curl jq python3; do
  command -v "$bin" >/dev/null || { echo "missing dependency: $bin" >&2; exit 1; }
done

# Reverse a Claude project slug (e.g.
# `-Users-me-src-acme-example-app`) into the
# real on-disk full path, by enumerating all 2^(n-1) /-vs-- splits
# between the n hyphen-separated parts and keeping the longest
# (most-specific) one that exists. Identical algorithm to
# import-claude-memory.sh's slug_to_basename but returns full path.
slug_to_path() {
  local slug="$1"
  slug="${slug#-}"
  local IFS='-'
  local -a parts
  read -r -a parts <<<"$slug"
  local n=${#parts[@]}
  if (( n == 1 )); then
    [[ -d "/${parts[0]}" ]] && printf '/%s' "${parts[0]}"
    return
  fi
  local max_mask=$(( (1 << (n-1)) - 1 ))
  local best=""
  local best_pathlen=0
  local mask i
  for ((mask=0; mask<=max_mask; mask++)); do
    local path="/${parts[0]}"
    for ((i=1; i<n; i++)); do
      if (( (mask >> (i-1)) & 1 )); then
        path+="-${parts[i]}"
      else
        path+="/${parts[i]}"
      fi
    done
    if [[ -d "$path" ]]; then
      if (( ${#path} > best_pathlen )); then
        best="$path"
        best_pathlen=${#path}
      fi
    fi
  done
  printf '%s' "$best"
}

# Enumerate matching files in $1 (repo root) using the manifest at $2.
# For each file emit ONE JSON line on stdout with:
#   abs_path, rel, project, size_bytes, content_hash, is_handoff,
#   handoff_date, chunk_group (sha8), chunks: [{index,total,heading,content}, ...]
#
# Chunking ladder is applied lazily — small files emit a single chunk
# with total=1 and heading=null; oversized files get the h2/h3/paragraph
# segmentation + greedy coalescing.
emit_files() {
  python3 - "$1" "$2" "$3" "$4" "$5" <<'PYEOF'
import os, sys, re, json, hashlib, fnmatch

repo_root      = sys.argv[1]
manifest_path  = sys.argv[2]
project        = sys.argv[3]
max_bytes      = int(sys.argv[4])
coalesce_ratio = float(sys.argv[5])
target_bytes   = max(1, int(max_bytes * coalesce_ratio))

includes, excludes = [], []
with open(manifest_path) as fh:
    for raw in fh:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!"):
            excludes.append(line[1:])
        else:
            includes.append(line)

# Segmented pattern matcher: ** matches zero or more path components;
# single-segment patterns work the same as plain fnmatch.fnmatchcase.
def match_pattern(rel, pat):
    return _match(rel.split("/"), pat.split("/"))

def _match(rps, pps):
    if not pps:
        return not rps
    p = pps[0]
    if p == "**":
        for i in range(0, len(rps) + 1):
            if _match(rps[i:], pps[1:]):
                return True
        return False
    if not rps:
        return False
    if fnmatch.fnmatchcase(rps[0], p):
        return _match(rps[1:], pps[1:])
    return False

PRUNE = {".git", "node_modules", "dist", "build", ".pnpm-store",
         "target", "venv", ".venv", ".next", ".turbo", "coverage"}

HANDOFF_RE = re.compile(r"session-handoff-(\d{4}-\d{2}-\d{2})", re.IGNORECASE)

# ----- chunking helpers -----

def split_by_heading(text, level):
    """Split text into sections by markdown heading of the given level.
    Returns list of (heading_text_or_None, body) tuples. Content before
    the first heading is emitted as (None, prefix)."""
    prefix_re = "#" * level + " "
    sections = []
    cur_head, cur_lines = None, []
    for line in text.split("\n"):
        if line.startswith(prefix_re):
            if cur_head is not None or cur_lines:
                sections.append((cur_head, "\n".join(cur_lines).rstrip()))
            cur_head = line[len(prefix_re):].strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_head is not None or cur_lines:
        sections.append((cur_head, "\n".join(cur_lines).rstrip()))
    return sections

def split_by_paragraph(text, budget):
    """Greedy-pack paragraphs into chunks of up to `budget` bytes each.
    A paragraph that alone exceeds budget is emitted as its own chunk
    (caller decides whether to skip it)."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks, cur = [], []
    cur_len = 0
    for p in paragraphs:
        plen = len(p.encode("utf-8")) + 2
        if cur and cur_len + plen > budget:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += plen
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks

def coalesce_sections(sections, budget):
    """Greedy-pack adjacent (heading, body) tuples until adding the next
    would exceed `budget`. Emits (representative_heading_or_None, body)
    tuples. Representative heading is the first heading in the pack, so
    chunk titles remain meaningful."""
    out = []
    cur_heads, cur_bodies = [], []
    cur_len = 0
    for head, body in sections:
        piece = (("## " + head + "\n\n") if head else "") + body
        plen = len(piece.encode("utf-8")) + 2
        if cur_bodies and cur_len + plen > budget:
            out.append((cur_heads[0] if cur_heads else None,
                        "\n\n".join(cur_bodies)))
            cur_heads, cur_bodies, cur_len = [], [], 0
        cur_heads.append(head)
        cur_bodies.append(piece)
        cur_len += plen
    if cur_bodies:
        out.append((cur_heads[0] if cur_heads else None,
                    "\n\n".join(cur_bodies)))
    return out

def chunk_content(text):
    """Return list of (heading_or_None, body) tuples ready to POST.
    Lossless: every byte of input survives somewhere in the output
    (modulo trailing whitespace trims). Body sizes capped at max_bytes
    with a coalesce target of target_bytes. Returns empty list only on
    empty input."""
    if len(text.encode("utf-8")) <= max_bytes:
        return [(None, text)]

    # ## split, recurse with ### if any chunk still oversized.
    h2 = split_by_heading(text, 2)
    if len(h2) > 1:
        refined = []
        for head, body in h2:
            piece_bytes = len(body.encode("utf-8")) + (len(head) + 4 if head else 0)
            if piece_bytes <= max_bytes:
                refined.append((head, body))
                continue
            # body too big — try ### inside
            h3 = split_by_heading(body, 3)
            if len(h3) > 1:
                for h3head, h3body in h3:
                    h3bytes = len(h3body.encode("utf-8"))
                    if h3bytes <= max_bytes:
                        # carry parent h2 as heading prefix so context isn't lost
                        joint = f"{head} / {h3head}" if (head and h3head) else (head or h3head)
                        refined.append((joint, h3body))
                    else:
                        # still too big — paragraph split
                        for para_chunk in split_by_paragraph(h3body, target_bytes):
                            joint = f"{head} / {h3head}" if (head and h3head) else (head or h3head)
                            refined.append((joint, para_chunk))
            else:
                # no ### either — paragraph split
                for para_chunk in split_by_paragraph(body, target_bytes):
                    refined.append((head, para_chunk))
        sections = refined
    else:
        # no ## — paragraph split from the start
        sections = [(None, p) for p in split_by_paragraph(text, target_bytes)]

    # Coalesce small adjacent sections up to target_bytes.
    coalesced = coalesce_sections(sections, target_bytes)

    # Last-chance guard: any chunk still > max_bytes gets skipped.
    final = []
    for head, body in coalesced:
        if len(body.encode("utf-8")) > max_bytes:
            sys.stderr.write(
                f"  WARN chunk too large after ladder ({len(body.encode('utf-8'))} bytes), skipping\n"
            )
            continue
        final.append((head, body))
    return final

# ----- file walk -----

candidates = []
for dirpath, dirnames, filenames in os.walk(repo_root):
    dirnames[:] = [d for d in dirnames if d not in PRUNE]
    for fn in filenames:
        abs_p = os.path.join(dirpath, fn)
        rel = os.path.relpath(abs_p, repo_root)
        if any(match_pattern(rel, p) for p in includes) and not any(
            match_pattern(rel, p) for p in excludes
        ):
            candidates.append((abs_p, rel))

candidates.sort(key=lambda x: x[1])

for abs_p, rel in candidates:
    try:
        with open(abs_p, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        sys.stderr.write(f"  WARN read failed {rel}: {e}\n")
        continue

    content_hash = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        sys.stderr.write(f"  WARN non-utf8 {rel}, skipping\n")
        continue

    basename = os.path.basename(abs_p)
    m = HANDOFF_RE.search(basename)
    is_handoff = m is not None
    handoff_date = m.group(1) if m else None

    chunk_group = hashlib.sha256(f"{project}:{rel}".encode("utf-8")).hexdigest()[:8]

    chunks_raw = chunk_content(text)
    chunks = []
    total = len(chunks_raw)
    for idx, (head, body) in enumerate(chunks_raw, start=1):
        chunks.append({
            "index":   idx,
            "total":   total,
            "heading": head,
            "content": body,
        })

    obj = {
        "abs_path":     abs_p,
        "rel":          rel,
        "project":      project,
        "size_bytes":   len(raw),
        "content_hash": content_hash,
        "is_handoff":   is_handoff,
        "handoff_date": handoff_date,
        "chunk_group":  chunk_group,
        "chunks":       chunks,
    }
    print(json.dumps(obj, ensure_ascii=False))
PYEOF
}

# ---------------------------------------------------------------------
# Load + migrate sidecar.
# Old V1 keys = absolute paths. New V2 keys = "<project>:<rel-path>".
# Migration: walk every Claude project, build repo paths, find each old
# key under one of them, rewrite to the new shape. Orphaned old keys
# (repo no longer on disk) are dropped with a warning.
# ---------------------------------------------------------------------
KNOWN_HASHES_JSON='{}'
if [[ "$CHANGED_ONLY" == 1 && -f "$HASH_FILE" ]]; then
  if KNOWN_HASHES_JSON=$(jq -c . <"$HASH_FILE" 2>/dev/null); then
    :
  else
    echo "warn: $HASH_FILE not valid JSON, treating as empty (re-imports will fire)" >&2
    KNOWN_HASHES_JSON='{}'
  fi
fi

migrate_sidecar() {
  local needs_migration
  needs_migration=$(jq -r '[keys[] | select(startswith("/"))] | length' <<<"$KNOWN_HASHES_JSON")
  [[ "$needs_migration" == "0" ]] && return 0

  local -a repos=()
  local proj_dir slug rp
  for proj_dir in "$PROJECTS_DIR"/*/; do
    slug=$(basename "$proj_dir")
    [[ "$slug" == *"claude-mem"* ]] && continue
    rp=$(slug_to_path "$slug")
    [[ -z "$rp" || ! -d "$rp" ]] && continue
    repos+=("$rp")
  done

  local migrated=0 orphaned=0 old_key matched matched_len h pb rel new_key
  while IFS= read -r old_key; do
    [[ -z "$old_key" ]] && continue
    matched=""; matched_len=0
    for rp in "${repos[@]}"; do
      if [[ "$old_key" == "$rp/"* && ${#rp} -gt $matched_len ]]; then
        matched="$rp"
        matched_len=${#rp}
      fi
    done
    h=$(jq -r --arg k "$old_key" '.[$k]' <<<"$KNOWN_HASHES_JSON")
    if [[ -n "$matched" ]]; then
      pb=$(basename "$matched")
      rel="${old_key#$matched/}"
      new_key="$pb:$rel"
      KNOWN_HASHES_JSON=$(jq -c --arg k "$old_key" --arg nk "$new_key" --arg h "$h" \
        'del(.[$k]) | .[$nk] = $h' <<<"$KNOWN_HASHES_JSON")
      migrated=$((migrated+1))
    else
      KNOWN_HASHES_JSON=$(jq -c --arg k "$old_key" 'del(.[$k])' <<<"$KNOWN_HASHES_JSON")
      orphaned=$((orphaned+1))
    fi
  done < <(jq -r 'keys[] | select(startswith("/"))' <<<"$KNOWN_HASHES_JSON")

  if (( migrated > 0 || orphaned > 0 )); then
    echo "sidecar migrated: ${migrated} key(s) converted, ${orphaned} orphaned (repo not found)" >&2
  fi
}

[[ "$CHANGED_ONLY" == 1 ]] && migrate_sidecar
NEW_HASHES_JSON="$KNOWN_HASHES_JSON"

# Liveness check upfront — no point hashing a corpus we can't POST to.

# ---- circuit breaker -------------------------------------------------
# A degraded server (loaded, mid-backfill, mid-migration) does not FAIL
# the per-file POST -- it just makes every one burn the full --max-time.
# At ~100 files that is 15-20 minutes of pointless work per invocation,
# and this script runs from a Stop hook, i.e. once per assistant turn.
# The livez pre-flight above does NOT protect against this: livez stays
# fast and returns 200 under exactly the load that makes the write
# endpoints slow, so it reports healthy while every import times out.
#
# So: count CONSECUTIVE timeouts/failures and abort the whole run once
# they cross the threshold. Nothing is lost -- hashes are recorded only
# on confirmed success, so the next --changed-only run retries whatever
# this one abandoned.
CIRCUIT_MAX_CONSECUTIVE="${AGENTMEMORY_IMPORT_CIRCUIT_MAX:-3}"
circuit_consecutive_failures=0
circuit_tripped=0

# Returns 0 while the circuit is closed, 1 once it has tripped.
circuit_ok() { [[ "$circuit_tripped" == 0 ]]; }

circuit_record() {
  # $1: "ok" or "fail"
  if [[ "$1" == ok ]]; then
    circuit_consecutive_failures=0
    return 0
  fi
  circuit_consecutive_failures=$((circuit_consecutive_failures + 1))
  if (( circuit_consecutive_failures >= CIRCUIT_MAX_CONSECUTIVE )); then
    circuit_tripped=1
    printf '  !!   circuit breaker: %d consecutive failures against %s — aborting this run (unrecorded files retry next time)\n' \
      "$circuit_consecutive_failures" "$URL" >&2
  fi
}
# ----------------------------------------------------------------------

if ! curl -fsS --connect-timeout 5 --max-time 10 "$URL/agentmemory/livez" >/dev/null; then
  echo "server not reachable at $URL" >&2
  exit 1
fi

echo "agentmemory repo-docs import — server: $URL"
[[ "$DRY_RUN" == 1 ]] && echo "DRY RUN: no POSTs will be made."
echo

processed=0
memories_ok=0
lessons_ok=0
unchanged=0
skipped_size=0
skipped_no_manifest=0
errors=0
handoffs_matched=0
handoffs_unmatched=0

# Sessions are fetched lazily — only when the first handoff file appears
# in the stream. SESSIONS_JSON stays "" until then.
SESSIONS_JSON=""

ensure_sessions_loaded() {
  [[ -n "$SESSIONS_JSON" ]] && return 0
  SESSIONS_JSON=$(curl -fsS --connect-timeout 5 --max-time 30 \
    "$URL/agentmemory/sessions" 2>&1 || echo '{"sessions":[]}')
  if ! jq -e '.sessions' >/dev/null 2>&1 <<<"$SESSIONS_JSON"; then
    echo "warn: failed to fetch /agentmemory/sessions, handoffs will not be session-tagged" >&2
    SESSIONS_JSON='{"sessions":[]}'
  fi
}

# Look up session IDs matching ($1 project basename, $2 YYYY-MM-DD).
# Pre-#475 sessions store .project as the full repo path; post-#475
# they store the basename. Normalize on the fly by taking the basename
# of whatever's there, then compare.
session_ids_for() {
  jq -r --arg p "$1" --arg d "$2" '
    .sessions[]
    | select(
        ((.project // "") | split("/") | .[-1]) == $p
        and ((.startedAt // "") | startswith($d))
      )
    | .id
  ' <<<"$SESSIONS_JSON"
}

# Walk one repo: read its manifest, stream files+chunks from python,
# hash-gate per file, POST per chunk.
process_repo() {
  local repo_path="$1"
  local project_basename
  project_basename="$(basename "$repo_path")"

  local manifest="$repo_path/.agentmemory-ingest"
  if [[ ! -f "$manifest" ]]; then
    skipped_no_manifest=$((skipped_no_manifest+1))
    return 0
  fi

  if [[ -n "$PROJECT_PATTERN" ]] && ! [[ "$project_basename" =~ $PROJECT_PATTERN ]]; then
    return 0
  fi

  local files_emitted=0
  printf '[%s] scanning...\n' "$project_basename"

  local line
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    files_emitted=$((files_emitted+1))

    local abs rel size content_hash is_handoff handoff_date chunk_group
    abs=$(jq -r '.abs_path' <<<"$line")
    rel=$(jq -r '.rel' <<<"$line")
    size=$(jq -r '.size_bytes' <<<"$line")
    content_hash=$(jq -r '.content_hash' <<<"$line")
    is_handoff=$(jq -r '.is_handoff' <<<"$line")
    handoff_date=$(jq -r '.handoff_date // ""' <<<"$line")
    chunk_group=$(jq -r '.chunk_group' <<<"$line")
    local key="$project_basename:$rel"

    if [[ "$LIMIT" -gt 0 && "$processed" -ge "$LIMIT" ]]; then
      break
    fi

    # Hash gate (per file, NOT per chunk). When the file changes, all
    # chunks re-POST — auto-supersede pairs them via the in-content
    # ordinal tokens.
    if [[ "$CHANGED_ONLY" == 1 ]]; then
      local prev_hash
      prev_hash=$(jq -r --arg k "$key" '.[$k] // empty' <<<"$KNOWN_HASHES_JSON")
      if [[ "$prev_hash" == "$content_hash" ]]; then
        unchanged=$((unchanged+1))
        continue
      fi
    fi

    # Resolve session tags for handoff files (lazy fetch).
    local -a session_tags=()
    if [[ "$is_handoff" == "true" && -n "$handoff_date" ]]; then
      ensure_sessions_loaded
      local sid
      while IFS= read -r sid; do
        [[ -z "$sid" ]] && continue
        session_tags+=("session:$sid")
      done < <(session_ids_for "$project_basename" "$handoff_date")
      if (( ${#session_tags[@]} > 0 )); then
        handoffs_matched=$((handoffs_matched+1))
      else
        handoffs_unmatched=$((handoffs_unmatched+1))
      fi
    fi

    # Iterate chunks and POST each. On any chunk failure, do NOT record
    # the hash — the whole file retries on the next run.
    local file_ok=1
    local chunks_total
    chunks_total=$(jq -r '.chunks | length' <<<"$line")

    if [[ "$DRY_RUN" == 1 ]]; then
      if [[ "$is_handoff" == "true" ]]; then
        printf '  [DRY] %s [HANDOFF date=%s sessions=%d chunks=%d] (%d bytes)\n' \
          "$rel" "$handoff_date" "${#session_tags[@]}" "$chunks_total" "$size"
      else
        printf '  [DRY] %s [chunks=%d] (%d bytes)\n' \
          "$rel" "$chunks_total" "$size"
      fi
      processed=$((processed+1))
      continue
    fi

    local i
    for ((i=0; i<chunks_total; i++)); do
      local idx total heading body content_prefix label
      idx=$(jq -r ".chunks[$i].index" <<<"$line")
      total=$(jq -r ".chunks[$i].total" <<<"$line")
      heading=$(jq -r ".chunks[$i].heading // \"\"" <<<"$line")
      body=$(jq -r ".chunks[$i].content" <<<"$line")

      # Build content header. Small docs (total==1) keep V1's tight
      # header; chunked docs include ordinal + group tokens that
      # Jaccard-pair-stably across re-imports.
      if [[ "$is_handoff" == "true" ]]; then
        label="[Session handoff]"
      else
        label="[Repo doc]"
      fi
      if (( total > 1 )); then
        if [[ -n "$heading" ]]; then
          content_prefix=$(printf '%s %s: %s (chunk %d/%d: %s, group %s:%d)' \
            "$label" "$project_basename" "$rel" "$idx" "$total" "$heading" "$chunk_group" "$idx")
        else
          content_prefix=$(printf '%s %s: %s (chunk %d/%d, group %s:%d)' \
            "$label" "$project_basename" "$rel" "$idx" "$total" "$chunk_group" "$idx")
        fi
      else
        content_prefix=$(printf '%s %s: %s' "$label" "$project_basename" "$rel")
      fi

      local content
      content=$(printf '%s\n\n%s' "$content_prefix" "$body")

      # Payload + endpoint diverge by route. Handoff chunks land as
      # project-scoped lessons (which feed reflect via KV.lessons) —
      # see plan notes on V3-A. Non-handoff chunks land as memories.
      local payload endpoint route_label sessiontags_array
      sessiontags_array=$(printf '%s\n' "${session_tags[@]+"${session_tags[@]}"}" \
        | jq -R . | jq -s .)

      if [[ "$is_handoff" == "true" ]]; then
        # Lesson tags: from:<rel>, session-handoff, project, session:<id>...
        # The from:<rel> tag survives chunking so all chunks of one file
        # are findable as a group. agentmemory's mem::lesson-save
        # fingerprints by content, so chunk N and chunk N+1 are distinct
        # lessons (different chunk-N/M tokens in content) — no collision.
        local tags_json
        tags_json=$(jq -nc \
          --arg fp "from:$rel" \
          --arg pj "$project_basename" \
          --argjson stags "$sessiontags_array" \
          '["session-handoff", $fp, $pj] + $stags')
        payload=$(jq -nc \
          --arg content "$content" \
          --arg project "$project_basename" \
          --argjson conf 0.9 \
          --argjson tags "$tags_json" \
          '{content:$content, project:$project, confidence:$conf, tags:$tags}')
        endpoint="$URL/agentmemory/lessons"
        route_label="LSN"
      else
        local concepts_json
        concepts_json=$(jq -nc --arg cg "chunk-group-$chunk_group" \
          --arg pj "$project_basename" \
          --argjson stags "$sessiontags_array" \
          '["repo-doc", $cg]')
        payload=$(jq -nc --arg content "$content" --argjson c "$concepts_json" \
          '{content:$content, type:"architecture", concepts:$c}')
        endpoint="$URL/agentmemory/remember"
        route_label="DOC"
      fi

      local resp
      if ! circuit_ok; then continue; fi
      resp=$(curl -sS --connect-timeout 5 --max-time 30 \
          -X POST "$endpoint" \
          -H 'content-type: application/json' \
          --data "$payload" 2>&1 || echo '{"success":false,"error":"curl_failed"}')

      if jq -e . >/dev/null 2>&1 <<<"$resp"; then
        local ok
        ok=$(jq -r '.success // .lesson.id // .memory.id // .id // "false"' <<<"$resp")
        if [[ "$ok" != "false" && "$ok" != "null" && -n "$ok" ]]; then
          if [[ "$is_handoff" == "true" ]]; then
            lessons_ok=$((lessons_ok+1))
          else
            memories_ok=$((memories_ok+1))
          fi
          if (( total > 1 )); then
            printf '  %s  %s [%d/%d]\n' "$route_label" "$rel" "$idx" "$total"
          else
            printf '  %s  %s\n' "$route_label" "$rel"
          fi
        else
          file_ok=0
          errors=$((errors+1))
          local err
          err=$(jq -r '.error // "unknown"' <<<"$resp")
          printf '  ERR  %s [%d/%d]: %s\n' "$rel" "$idx" "$total" "$err"
        fi
      else
        file_ok=0
        errors=$((errors+1))
        printf '  ERR  %s [%d/%d]: %s\n' "$rel" "$idx" "$total" "$(head -c 200 <<<"$resp")"
      fi
    done

    # Only record the hash if every chunk succeeded.
    if [[ "$CHANGED_ONLY" == 1 && "$file_ok" == 1 ]]; then
      NEW_HASHES_JSON=$(jq -c --arg k "$key" --arg h "$content_hash" \
        '.[$k] = $h' <<<"$NEW_HASHES_JSON")
    fi

    processed=$((processed+1))
  done < <(emit_files "$repo_path" "$manifest" "$project_basename" "$MAX_BYTES" "$COALESCE_TARGET")

  printf '[%s] %d file(s) processed\n' "$project_basename" "$files_emitted"
}

# Source: explicit --repo wins; otherwise enumerate via Claude projects.
if [[ -n "$EXPLICIT_REPO" ]]; then
  [[ -d "$EXPLICIT_REPO" ]] || { echo "not a directory: $EXPLICIT_REPO" >&2; exit 1; }
  process_repo "$EXPLICIT_REPO"
else
  for proj_dir in "$PROJECTS_DIR"/*/; do
    slug=$(basename "$proj_dir")
    [[ "$slug" == *"claude-mem"* ]] && continue  # observer junk
    repo_path=$(slug_to_path "$slug")
    [[ -z "$repo_path" || ! -d "$repo_path" ]] && continue
    process_repo "$repo_path"
  done
fi

# Atomic sidecar write — only when --changed-only and not dry-run.
if [[ "$CHANGED_ONLY" == 1 && "$DRY_RUN" == 0 ]]; then
  mkdir -p "$(dirname "$HASH_FILE")"
  tmp="${HASH_FILE}.tmp.$$"
  printf '%s\n' "$NEW_HASHES_JSON" | jq . >"$tmp"
  mv "$tmp" "$HASH_FILE"
fi

echo
printf 'summary: processed=%d memories=%d lessons=%d unchanged=%d handoffs_matched=%d handoffs_unmatched=%d skipped_no_manifest=%d errors=%d\n' \
  "$processed" "$memories_ok" "$lessons_ok" "$unchanged" "$handoffs_matched" "$handoffs_unmatched" "$skipped_no_manifest" "$errors"
