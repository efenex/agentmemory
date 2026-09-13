#!/usr/bin/env bash
# Import Claude Code's auto-memory (~/.claude/projects/*/memory/) into agentmemory.
#
# Each project under ~/.claude/projects/ has a memory/ subdirectory with
# markdown files Claude wrote about that project. The filename prefix
# carries the type:
#
#   feedback_*.md  →  Lesson (project-scoped, confidence 0.9)
#   project_*.md   →  Memory (type=architecture, global)
#   reference_*.md →  Memory (type=fact, global)
#   user_*.md      →  Memory (type=preference, global)
#   MEMORY.md      →  skip (index file, no content)
#
# Notable upstream constraint: agentmemory's Memory records (mem::remember)
# do NOT carry a project field — they're global. Only lessons can be
# project-scoped. project_*.md and reference_*.md therefore land global.
# That's fine: smart-search returns them regardless of project scope.
#
# Idempotency:
#   - Lessons: mem::lesson-save auto-strengthens on duplicate content,
#     so re-runs bump confidence rather than creating duplicates.
#   - Memories: mem::remember auto-supersedes on Jaccard similarity
#     > 0.7, so re-runs create a new version that supersedes the old
#     (preserving history rather than fragmenting it).
#
# --changed-only mode (for Stop-hook use):
#   Without a gate, re-running on every session-stop would re-strengthen
#   every file forever — decay/forgetting would never engage. So when
#   --changed-only is set we record a per-file SHA-256 in a sidecar at
#   ~/.agentmemory/.import-hashes.json and skip files whose content
#   hasn't changed since the last successful import. Hash entries are
#   updated only on successful POST (a failed import retries next run).
#
# Cost: $0. All imports use existing local agentmemory endpoints; no
# LLM provider involvement.
#
# Usage:
#   scripts/import-claude-memory.sh --dry-run
#   scripts/import-claude-memory.sh --project-pattern 'gitops-(assistant|caro)'
#   scripts/import-claude-memory.sh --changed-only  # only files whose
#                                                   # content has changed
#                                                   # since last import
#   scripts/import-claude-memory.sh                 # everything (re-strengthens)

set -euo pipefail

URL="${AGENTMEMORY_URL:-http://localhost:3111}"
PROJECTS_DIR="${CLAUDE_PROJECTS_DIR:-$HOME/.claude/projects}"
HASH_FILE="${AGENTMEMORY_IMPORT_HASHES:-$HOME/.agentmemory/.import-hashes.json}"
DRY_RUN=0
LIMIT=0
PROJECT_PATTERN=""
INCLUDE_OBSERVER=0  # skip claude-mem-observer-sessions by default
CHANGED_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)          DRY_RUN=1; shift ;;
    --limit)            LIMIT="${2:?--limit needs a number}"; shift 2 ;;
    --project-pattern)  PROJECT_PATTERN="${2:?--project-pattern needs a regex}"; shift 2 ;;
    --include-observer) INCLUDE_OBSERVER=1; shift ;;
    --changed-only)     CHANGED_ONLY=1; shift ;;
    -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

for bin in curl jq; do
  command -v "$bin" >/dev/null || { echo "missing dependency: $bin" >&2; exit 1; }
done

# Portable sha256 of a file's contents.
hash_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

# Load existing hash sidecar as a single jq value. {} when absent or
# unparseable — the latter triggers a warning rather than silent loss so
# we don't drop the index from a stray write.
KNOWN_HASHES_JSON='{}'
if [[ "$CHANGED_ONLY" == 1 && -f "$HASH_FILE" ]]; then
  if KNOWN_HASHES_JSON=$(jq -c . <"$HASH_FILE" 2>/dev/null); then
    :
  else
    echo "warn: $HASH_FILE not valid JSON, treating as empty (re-imports will fire)" >&2
    KNOWN_HASHES_JSON='{}'
  fi
fi
# Updated map — starts as a copy of known, gets entries overwritten on
# successful import. Files that error out are left at their previous
# hash (or unset) so the next run retries them.
NEW_HASHES_JSON="$KNOWN_HASHES_JSON"

# Reverse Claude Code's project-dir slug back to a basename.
# Slug format: leading `-` then path with `/` replaced by `-`.
#
# Ambiguity: real path components can themselves contain hyphens
# (e.g. /Users/me/src/acme/example-app has
# TWO hyphenated components). The slug
# -Users-me-src-acme-example-app offers no
# way to disambiguate from the string alone.
#
# Resolution: enumerate all 2^(n-1) possible /-vs-- splits between the
# n hyphen-separated parts, keep the ones whose constructed path
# actually exists on disk, return the basename of the longest such path
# (most specific match wins). 2^14 ≈ 16k checks is fine for slugs of
# realistic depth (~8 parts → 128 checks).
slug_to_basename() {
  local slug="$1"
  slug="${slug#-}"
  local IFS='-'
  local -a parts
  read -r -a parts <<<"$slug"
  local n=${#parts[@]}
  if (( n == 1 )); then
    printf '%s' "${parts[0]}"
    return
  fi
  local max_mask=$(( (1 << (n-1)) - 1 ))
  local best=""
  local best_pathlen=0
  local mask i
  for ((mask=0; mask<=max_mask; mask++)); do
    local path="/${parts[0]}"
    for ((i=1; i<n; i++)); do
      # bit i-1 set = keep as hyphen, unset = slash separator
      if (( (mask >> (i-1)) & 1 )); then
        path+="-${parts[i]}"
      else
        path+="/${parts[i]}"
      fi
    done
    if [[ -d "$path" ]]; then
      # Tiebreak: prefer longer paths (= more specific match).
      if (( ${#path} > best_pathlen )); then
        best="${path##*/}"
        best_pathlen=${#path}
      fi
    fi
  done
  if [[ -n "$best" ]]; then
    printf '%s' "$best"
  else
    # No real path matched — repo moved, deleted, or never on this
    # machine. Fall back to the rightmost hyphen-segment so we still
    # tag the import with *something* searchable.
    printf '%s' "${parts[n-1]}"
  fi
}

# Extract a single frontmatter field. YAML is simple here: `key: value`
# on one line within the --- fences. We don't handle multiline values
# because the auto-memory files don't use them.
extract_field() {
  local file="$1" key="$2"
  awk -v key="$key" '
    BEGIN { in_fm=0 }
    /^---$/ { in_fm++; if (in_fm==2) exit; next }
    in_fm==1 && $0 ~ "^"key": " { sub("^"key": ", ""); print; exit }
  ' "$file"
}

# Extract everything after the closing --- fence.
extract_body() {
  awk '
    BEGIN { in_fm=0; print_now=0 }
    /^---$/ { in_fm++; if (in_fm==2) { print_now=1; next } }
    print_now { print }
  ' "$1"
}

# Liveness check

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

echo "agentmemory claude-memory import — server: $URL"
[[ "$DRY_RUN" == 1 ]] && echo "DRY RUN: no POSTs will be made."
echo "source: $PROJECTS_DIR"
echo

processed=0
lessons_ok=0
memories_ok=0
skipped=0
unchanged=0
errors=0

for proj_dir in "$PROJECTS_DIR"/*/; do
  [[ -d "$proj_dir/memory" ]] || continue
  slug=$(basename "$proj_dir")

  if [[ "$INCLUDE_OBSERVER" == 0 && "$slug" == *"claude-mem"* ]]; then
    continue
  fi

  project=$(slug_to_basename "$slug")

  if [[ -n "$PROJECT_PATTERN" ]] && ! [[ "$project" =~ $PROJECT_PATTERN ]]; then
    continue
  fi

  file_count=$(find "$proj_dir/memory" -maxdepth 1 -name "*.md" | wc -l | tr -d ' ')
  [[ "$file_count" == "0" ]] && continue
  printf '[%s] %d files\n' "$project" "$file_count"

  for file in "$proj_dir/memory"/*.md; do
    [[ -f "$file" ]] || continue
    filename=$(basename "$file")
    [[ "$filename" == "MEMORY.md" ]] && continue

    if [[ "$LIMIT" -gt 0 && "$processed" -ge "$LIMIT" ]]; then
      break 2
    fi

    if [[ "$CHANGED_ONLY" == 1 ]]; then
      current_hash=$(hash_file "$file")
      prev_hash=$(jq -r --arg p "$file" '.[$p] // empty' <<<"$KNOWN_HASHES_JSON")
      if [[ "$prev_hash" == "$current_hash" ]]; then
        unchanged=$((unchanged+1))
        continue
      fi
    else
      current_hash=""
    fi

    name=$(extract_field "$file" "name")
    description=$(extract_field "$file" "description")
    type=$(extract_field "$file" "type")
    body=$(extract_body "$file")

    # Fallback to filename prefix when frontmatter type is missing.
    if [[ -z "$type" ]]; then
      type="${filename%%_*}"
    fi

    # Combine name + body so content is fully searchable.
    if [[ -n "$name" ]]; then
      content=$(printf '%s\n\n%s' "$name" "$body")
    else
      content="$body"
    fi

    if [[ "$DRY_RUN" == 1 ]]; then
      printf '  [DRY] %-10s %s\n' "$type" "$filename"
      processed=$((processed+1))
      continue
    fi

    case "$type" in
      feedback)
        payload=$(jq -nc \
          --arg content "$content" \
          --arg context "$description" \
          --arg project "$project" \
          --argjson conf 0.9 \
          --arg tag1 "claude-mem-import" \
          --arg tag2 "from:$filename" \
          '{content:$content, context:$context, project:$project, confidence:$conf, tags:[$tag1,$tag2]}')
        endpoint="$URL/agentmemory/lessons"
        ;;
      project|architecture|pattern)
        payload=$(jq -nc \
          --arg content "$content" \
          '{content:$content, type:"architecture"}')
        endpoint="$URL/agentmemory/remember"
        ;;
      reference|fact)
        payload=$(jq -nc \
          --arg content "$content" \
          '{content:$content, type:"fact"}')
        endpoint="$URL/agentmemory/remember"
        ;;
      user|preference)
        payload=$(jq -nc \
          --arg content "$content" \
          '{content:$content, type:"preference"}')
        endpoint="$URL/agentmemory/remember"
        ;;
      workflow)
        payload=$(jq -nc \
          --arg content "$content" \
          '{content:$content, type:"workflow"}')
        endpoint="$URL/agentmemory/remember"
        ;;
      *)
        printf '  ??   %s (unknown type=%s)\n' "$filename" "$type"
        skipped=$((skipped+1))
        processed=$((processed+1))
        continue
        ;;
    esac

    if ! circuit_ok; then
      skipped=$((skipped+1)); processed=$((processed+1)); continue
    fi
    resp=$(curl -sS --connect-timeout 5 --max-time 10 \
        -X POST "$endpoint" \
        -H 'content-type: application/json' \
        --data "$payload" 2>&1 || echo '{"success":false,"error":"curl_failed"}')
    if grep -q '"error":"curl_failed"' <<<"$resp"; then circuit_record fail; else circuit_record ok; fi

    if jq -e . >/dev/null 2>&1 <<<"$resp"; then
      ok=$(jq -r '.success // .lesson.id // .memory.id // .id // "false"' <<<"$resp")
      if [[ "$ok" != "false" && "$ok" != "null" && -n "$ok" ]]; then
        if [[ "$type" == "feedback" ]]; then
          lessons_ok=$((lessons_ok+1))
          printf '  LSN  %s\n' "$filename"
        else
          memories_ok=$((memories_ok+1))
          printf '  MEM  %-10s %s\n' "$type" "$filename"
        fi
        # Record the hash only on confirmed success; failed imports
        # stay un-recorded so the next --changed-only run retries them.
        if [[ "$CHANGED_ONLY" == 1 && -n "$current_hash" ]]; then
          NEW_HASHES_JSON=$(jq -c --arg p "$file" --arg h "$current_hash" \
            '.[$p] = $h' <<<"$NEW_HASHES_JSON")
        fi
      else
        errors=$((errors+1))
        err=$(jq -r '.error // "unknown"' <<<"$resp")
        printf '  ERR  %s: %s\n' "$filename" "$err"
      fi
    else
      errors=$((errors+1))
      printf '  ERR  %s: %s\n' "$filename" "$(head -c 200 <<<"$resp")"
    fi

    processed=$((processed+1))
  done
done

# Atomic-write the updated sidecar (only in --changed-only mode, only
# when not dry-running). Failure here is fatal — silent corruption of
# the index would be worse than a noisy exit.
if [[ "$CHANGED_ONLY" == 1 && "$DRY_RUN" == 0 ]]; then
  mkdir -p "$(dirname "$HASH_FILE")"
  tmp="${HASH_FILE}.tmp.$$"
  printf '%s\n' "$NEW_HASHES_JSON" | jq . >"$tmp"
  mv "$tmp" "$HASH_FILE"
fi

echo
printf 'summary: processed=%d lessons=%d memories=%d skipped=%d unchanged=%d errors=%d\n' \
  "$processed" "$lessons_ok" "$memories_ok" "$skipped" "$unchanged" "$errors"
