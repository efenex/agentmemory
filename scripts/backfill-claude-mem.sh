#!/usr/bin/env bash
# Backfill claude-mem (~/.claude-mem/claude-mem.db) into agentmemory.
#
# Reads claude-mem's SQLite database, transforms each row into
# agentmemory's Session / CompressedObservation / SessionSummary shape,
# and POSTs to /agentmemory/backfill/*. The backfill endpoint family is
# gated by AGENTMEMORY_BACKFILL_ENABLED=true on the daemon side.
#
# Every imported record is tagged with the concept `claude-mem-backfill`
# so the import is grep-able and forget-able as a group later.
#
# Usage:
#   ./scripts/backfill-claude-mem.sh --dry-run
#       Show what would be imported without writing anything.
#   ./scripts/backfill-claude-mem.sh
#       Actually push to AGENTMEMORY_URL (default http://127.0.0.1:3111).
#
# Requires: sqlite3 (with JSON mode, ≥3.33), jq, curl.

set -euo pipefail

DB="${CLAUDE_MEM_DB:-${HOME}/.claude-mem/claude-mem.db}"
AM_URL="${AGENTMEMORY_URL:-http://127.0.0.1:3111}"
DRY_RUN=0
MARKER_CONCEPT="claude-mem-backfill"

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

if [ ! -r "$DB" ]; then
  echo "claude-mem db not readable: $DB" >&2
  exit 1
fi

for cmd in sqlite3 jq curl; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "missing required command: $cmd" >&2
    exit 1
  fi
done

echo "[backfill] source : $DB"
echo "[backfill] target : $AM_URL"
echo "[backfill] dry-run: $DRY_RUN"

# Verify the daemon is reachable + the backfill endpoint is enabled
# before we start grinding through 800+ rows.
if [ "$DRY_RUN" -eq 0 ]; then
  http_code=$(curl -sS -o /tmp/backfill-probe.json -w "%{http_code}" \
    -X POST "$AM_URL/agentmemory/backfill/session" \
    -H 'Content-Type: application/json' \
    -d '{"sessionId":"","project":""}' || echo 000)
  if [ "$http_code" = "503" ]; then
    echo "[backfill] daemon returns 503 — set AGENTMEMORY_BACKFILL_ENABLED=true in the env, restart the daemon, retry." >&2
    cat /tmp/backfill-probe.json >&2
    exit 1
  fi
  if [ "$http_code" = "000" ]; then
    echo "[backfill] could not reach $AM_URL — is the daemon up?" >&2
    exit 1
  fi
  # 400 = endpoint live + reachable, our empty payload was rejected on validation.
  if [ "$http_code" != "400" ]; then
    echo "[backfill] unexpected probe response: HTTP $http_code" >&2
    cat /tmp/backfill-probe.json >&2
    exit 1
  fi
  echo "[backfill] daemon reachable + backfill endpoint enabled."
fi

post() {
  local path="$1"
  local body="$2"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY $path :: $(echo "$body" | jq -c '{sessionId, id, title, project} | with_entries(select(.value != null))')"
    return 0
  fi
  local code
  code=$(curl -sS -o /tmp/backfill-resp.json -w "%{http_code}" \
    -X POST "$AM_URL$path" \
    -H 'Content-Type: application/json' \
    -d "$body")
  if [ "$code" -ge 400 ]; then
    echo "[backfill] FAIL $path HTTP $code body=$body" >&2
    cat /tmp/backfill-resp.json >&2
    return 1
  fi
}

# ---------- sessions ----------
# Only sessions with a non-empty project AND attached observations or
# summaries get imported. The 12 empty-project + 5 empty-data
# single-session projects are noise.
echo ""
echo "==> Importing sessions"
session_count=0
while IFS= read -r row; do
  session_id=$(echo "$row" | jq -r '.memory_session_id')
  project=$(echo "$row" | jq -r '.project')
  if [ -z "$session_id" ] || [ "$session_id" = "null" ]; then continue; fi
  if [ -z "$project" ] || [ "$project" = "null" ]; then continue; fi

  body=$(echo "$row" | jq --arg marker "$MARKER_CONCEPT" '{
    sessionId: .memory_session_id,
    project: .project,
    cwd: .project,
    startedAt: .started_at,
    endedAt: .completed_at,
    status: (if .status == "active" then "active" elif .status == "failed" then "abandoned" else "completed" end),
    firstPrompt: .user_prompt,
    summary: .custom_title,
    tags: [$marker]
  }')
  post /agentmemory/backfill/session "$body"
  session_count=$((session_count + 1))
done < <(sqlite3 "$DB" -json "
  SELECT s.memory_session_id, s.project, s.started_at, s.completed_at,
         s.status, s.user_prompt, s.custom_title
  FROM sdk_sessions s
  WHERE s.project != '' AND s.project IS NOT NULL
    AND s.memory_session_id IS NOT NULL
    AND (
      EXISTS (SELECT 1 FROM observations o WHERE o.memory_session_id = s.memory_session_id)
      OR EXISTS (SELECT 1 FROM session_summaries ss WHERE ss.memory_session_id = s.memory_session_id)
    )
  ORDER BY s.started_at_epoch
" | jq -c '.[]')
echo "[backfill] sessions imported: $session_count"

# ---------- observations ----------
echo ""
echo "==> Importing observations"
obs_count=0
while IFS= read -r row; do
  body=$(echo "$row" | jq --arg marker "$MARKER_CONCEPT" '
    {
      id: ("obs_backfill_" + (.id | tostring)),
      sessionId: .memory_session_id,
      timestamp: .created_at,
      type: (if (.type // "") | test("file_(read|write|edit)|command_run|search|web_fetch|conversation|error|decision|discovery|subagent|notification|task|image"; "x") then .type else "other" end),
      title: (.title // ((.text // "untitled") | .[0:120])),
      subtitle: .subtitle,
      facts: (try (.facts | fromjson) catch (if .facts then [.facts] else [] end)),
      narrative: (.narrative // .text // ""),
      concepts: (((try (.concepts | fromjson) catch (if .concepts then [.concepts] else [] end)) + [$marker]) | unique),
      files: (((try (.files_modified | fromjson) catch []) + (try (.files_read | fromjson) catch [])) | unique),
      importance: 5,
      confidence: 0.5
    }
    | with_entries(select(.value != null))
  ')
  post /agentmemory/backfill/observation "$body"
  obs_count=$((obs_count + 1))
done < <(sqlite3 "$DB" -json "
  SELECT id, memory_session_id, project, text, type, title, subtitle,
         facts, narrative, concepts, files_read, files_modified,
         created_at
  FROM observations
  WHERE project != '' AND project IS NOT NULL
  ORDER BY created_at_epoch
" | jq -c '.[]')
echo "[backfill] observations imported: $obs_count"

# ---------- summaries ----------
echo ""
echo "==> Importing session summaries"
sum_count=0
while IFS= read -r row; do
  body=$(echo "$row" | jq --arg marker "$MARKER_CONCEPT" '
    {
      sessionId: .memory_session_id,
      project: .project,
      createdAt: .created_at,
      title: ((.request // "untitled") | .[0:200]),
      narrative: (
        [(.request // ""), (.investigated // ""), (.learned // ""), (.notes // "")]
        | map(select(length > 0))
        | join("\n\n")
      ),
      keyDecisions: (
        try ((.completed // "") | split("\n") | map(select(length > 0)))
        catch []
      ),
      filesModified: (
        ((try (.files_edited | fromjson) catch []) + (try (.files_read | fromjson) catch []))
        | unique
      ),
      concepts: [$marker],
      observationCount: 0
    }
    | with_entries(select(.value != null))
  ')
  post /agentmemory/backfill/summary "$body"
  sum_count=$((sum_count + 1))
done < <(sqlite3 "$DB" -json "
  SELECT memory_session_id, project, request, investigated, learned,
         completed, next_steps, files_read, files_edited, notes,
         created_at
  FROM session_summaries
  WHERE project != '' AND project IS NOT NULL
  ORDER BY created_at_epoch
" | jq -c '.[]')
echo "[backfill] summaries imported: $sum_count"

echo ""
echo "[backfill] done."
echo "  sessions     : $session_count"
echo "  observations : $obs_count"
echo "  summaries    : $sum_count"
echo "  marker       : concept = '$MARKER_CONCEPT' (grep-able in viewer)"
if [ "$DRY_RUN" -eq 0 ]; then
  echo ""
  echo "Next: run scripts/rebuild-graph.sh if you want the imported"
  echo "observations indexed into the BM25 + vector + graph stores."
fi
