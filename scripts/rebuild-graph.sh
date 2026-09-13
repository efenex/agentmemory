#!/usr/bin/env bash
# rebuild-graph.sh — one-shot full graph-extraction pass over agentmemory
# content. Pointed at a local LLM endpoint, the cost is negligible.
#
# Background: mem::remember does NOT auto-trigger mem::graph-extract,
# so Memory records (e.g., V2 repo-doc imports, post-hoc lessons, etc.)
# never populate the knowledge-graph layer that reflect clusters on.
# Observations *do* auto-trigger extraction when GRAPH_EXTRACTION_ENABLED
# is on, but only at write-time — content imported before the flag was
# set is never back-filled. This script closes both gaps with a single
# bulk sweep.
#
# What it does:
#   1. If --with-endpoint is given, save current OPENAI_* from
#      ~/.agentmemory-docker/.env, append override values, force-recreate
#      the container so the daemon picks up the local-LLM endpoint. An
#      EXIT trap restores the original env + recreates the container
#      whether the script succeeds, errors, or is killed.
#   2. Wait for /livez.
#   3. Iterate selected content types (memories, lessons, summaries,
#      observations) via REST. Synthesize a CompressedObservation-shaped
#      wrapper per item.
#   4. POST batches of N items at a time to /agentmemory/graph-extract.
#      Agentmemory dedupes graph nodes by (name, type) and reinforces
#      existing edges, so this is safe to re-run.
#   5. Log per-batch progress + timing.
#
# Cost: scales with --types selected and --batch-size. With a local LLM
# (vLLM / LM Studio / Ollama), one full sweep over a few hundred items
# is essentially free. With the default serverless LLM (DeepSeek V4-Flash
# on Novita) it's a real bill — refuse to run without --with-endpoint
# unless --force-current-endpoint is passed.
#
# Usage:
#   scripts/rebuild-graph.sh --with-endpoint http://localhost:8000 \
#                            --api-key dummy --model qwen3
#   scripts/rebuild-graph.sh --with-endpoint https://my-vllm \
#                            --api-key vllm-no-auth --model qwen3.6 \
#                            --concurrency 4     # 4 batches in-flight at once
#                            --insecure-tls      # self-signed cert on the LLM
#                            --disable-thinking  # Qwen3/GLM/Kimi/DS-V4: kill <think>
#   scripts/rebuild-graph.sh --with-endpoint <url> --types memory,lesson
#   scripts/rebuild-graph.sh --with-endpoint <url> --project gitops --dry-run
#   scripts/rebuild-graph.sh --force-current-endpoint --types memory \
#                            --limit 5                # smoke-test the
#                                                     # daemon's current
#                                                     # serverless LLM
#
# --batch-size vs --concurrency:
#   --batch-size N    : N observations packed into ONE LLM call
#                       (bigger prompt, single response)
#   --concurrency M   : M LLM calls in-flight at once
#                       (vLLM batches them server-side at the GPU)
#   Total throughput  : roughly M × (one-batch latency); start at
#                       --concurrency 4 for self-hosted vLLM and tune
#                       up until you see queue-induced timeouts.
#
# IMPORTANT URL SHAPE: agentmemory's OpenAI-compat LLM provider always
# appends `/v1/chat/completions` to OPENAI_BASE_URL. So pass the host
# root WITHOUT a trailing `/v1` — e.g. `https://vllm/` not
# `https://vllm/v1`. If you pass `…/v1` the actual call becomes
# `…/v1/v1/chat/completions` and the server returns 404.

set -euo pipefail

URL="${AGENTMEMORY_URL:-http://localhost:3111}"
ENV_FILE="${AGENTMEMORY_ENV_FILE:-$HOME/.agentmemory-docker/.env}"
COMPOSE_FILE="${AGENTMEMORY_COMPOSE_FILE:-$(cd "$(dirname "$0")/.." && pwd)/docker/docker-compose.yml}"

WITH_ENDPOINT=""
API_KEY=""
MODEL=""
TYPES="memory,lesson,summary,observation"
PROJECT_PATTERN=""
DRY_RUN=0
LIMIT=0
BATCH_SIZE="${AGENTMEMORY_GRAPH_BATCH:-5}"
CONCURRENCY="${AGENTMEMORY_GRAPH_CONCURRENCY:-1}"
FORCE_CURRENT_ENDPOINT=0
INSECURE_TLS=0
DISABLE_THINKING=0

usage() {
  sed -n '2,46p' "$0"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-endpoint)          WITH_ENDPOINT="${2:?--with-endpoint needs a URL}"; shift 2 ;;
    --api-key)                API_KEY="${2:?--api-key needs a value}"; shift 2 ;;
    --model)                  MODEL="${2:?--model needs a value}"; shift 2 ;;
    --types)                  TYPES="${2:?--types needs a value}"; shift 2 ;;
    --project)                PROJECT_PATTERN="${2:?--project needs a regex}"; shift 2 ;;
    --dry-run)                DRY_RUN=1; shift ;;
    --limit)                  LIMIT="${2:?--limit needs a number}"; shift 2 ;;
    --batch-size)             BATCH_SIZE="${2:?--batch-size needs a number}"; shift 2 ;;
    --concurrency)            CONCURRENCY="${2:?--concurrency needs a number}"; shift 2 ;;
    --force-current-endpoint) FORCE_CURRENT_ENDPOINT=1; shift ;;
    --insecure-tls)           INSECURE_TLS=1; shift ;;
    --disable-thinking)       DISABLE_THINKING=1; shift ;;
    -h|--help)                usage 0 ;;
    *) echo "unknown flag: $1" >&2; usage 2 ;;
  esac
done

for bin in curl jq docker; do
  command -v "$bin" >/dev/null || { echo "missing dependency: $bin" >&2; exit 1; }
done

if [[ -z "$WITH_ENDPOINT" && "$FORCE_CURRENT_ENDPOINT" == 0 ]]; then
  cat >&2 <<EOF
Refusing to run against the daemon's current LLM endpoint. Bulk graph
extraction will burn through tokens at the serverless provider. Either:

  --with-endpoint http://localhost:<port>/v1   (recommended: local LLM)
  --force-current-endpoint                     (override, accept the bill)
EOF
  exit 1
fi

# ----- env swap + restore (only when --with-endpoint) -----

ENV_BACKUP=""
ENV_MARKER_START="# === rebuild-graph.sh override (auto-restored on script exit) ==="
ENV_MARKER_END="# === end override ==="

restore_env() {
  # Always runs on EXIT (success/error/signal). Safe no-op if no swap.
  if [[ -z "$ENV_BACKUP" ]]; then
    return 0
  fi
  if [[ -f "$ENV_BACKUP" ]]; then
    echo "[rebuild-graph] restoring $ENV_FILE from backup" >&2
    cp "$ENV_BACKUP" "$ENV_FILE"
    rm -f "$ENV_BACKUP"
  fi
  # Recreate container with the restored env. Non-fatal if it fails —
  # the user can recover manually with `docker compose up -d --force-recreate`.
  docker compose -f "$COMPOSE_FILE" up -d --force-recreate >/dev/null 2>&1 || \
    echo "[rebuild-graph] WARNING: failed to force-recreate; restore manually" >&2
}

apply_endpoint_swap() {
  ENV_BACKUP="${ENV_FILE}.rebuild-graph-backup.$$"
  cp "$ENV_FILE" "$ENV_BACKUP"
  {
    echo ""
    echo "$ENV_MARKER_START"
    echo "OPENAI_BASE_URL=$WITH_ENDPOINT"
    [[ -n "$API_KEY" ]] && echo "OPENAI_API_KEY=$API_KEY"
    [[ -n "$MODEL"  ]] && echo "OPENAI_MODEL=$MODEL"
    # Process-wide TLS verify disable. Bounded to this container during
    # this script run; restored by the EXIT trap. Use when the
    # --with-endpoint server uses a self-signed cert (e.g. self-hosted
    # vLLM behind an OpenShift router with its own CA).
    [[ "$INSECURE_TLS" == 1 ]] && echo "NODE_TLS_REJECT_UNAUTHORIZED=0"
    # AGENTMEMORY_DISABLE_THINKING signals the OpenAI LLM provider to
    # send chat_template_kwargs.enable_thinking=false and prepend
    # /no_think to system messages. Required for Qwen3-family models
    # via vLLM with --reasoning-parser; without it, the response's
    # actual content lands in `reasoning` and structured-output prompts
    # (graph extraction!) truncate inside the thinking block.
    [[ "$DISABLE_THINKING" == 1 ]] && echo "AGENTMEMORY_DISABLE_THINKING=true"
    echo "$ENV_MARKER_END"
  } >> "$ENV_FILE"
  echo "[rebuild-graph] env swapped — OPENAI_BASE_URL=$WITH_ENDPOINT (override appended to $ENV_FILE; will be restored on exit)"
  docker compose -f "$COMPOSE_FILE" up -d --force-recreate >/dev/null
  echo "[rebuild-graph] container recreated, waiting for /livez..."
  for i in $(seq 1 30); do
    if curl -fsS --max-time 3 "$URL/agentmemory/livez" >/dev/null 2>&1; then
      echo "[rebuild-graph] /livez ok"
      return 0
    fi
    sleep 1
  done
  echo "[rebuild-graph] /livez did not respond within 30s, aborting" >&2
  exit 1
}

# PIDs of currently-running background graph-extract workers. The EXIT
# handler kills any survivors so a Ctrl-C doesn't leave detached curls
# hammering the LLM.
INFLIGHT_PIDS=()

# All array-iteration helpers below guard against empty arrays
# explicitly: macOS bash 3.2 errors on `"${arr[@]}"` under `set -u`
# when arr is empty, and the `${arr[@]+expansion}` indirection is
# easy to mis-quote in ways that bite later. Pre-checking ${#arr[@]}
# is verbose but bulletproof.

kill_inflight() {
  if [[ ${#INFLIGHT_PIDS[@]} -gt 0 ]]; then
    local pid
    for pid in "${INFLIGHT_PIDS[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
  fi
  INFLIGHT_PIDS=()
}

on_exit() {
  kill_inflight
  restore_env
}

if [[ -n "$WITH_ENDPOINT" ]]; then
  [[ -f "$ENV_FILE" ]] || { echo "env file not found: $ENV_FILE" >&2; exit 1; }
  trap on_exit EXIT
  apply_endpoint_swap
else
  trap kill_inflight EXIT
fi

# Liveness check (in case --force-current-endpoint and no swap).
if ! curl -fsS --max-time 5 "$URL/agentmemory/livez" >/dev/null; then
  echo "server not reachable at $URL" >&2
  exit 1
fi

# ----- per-content-type observation synthesis -----
# Each helper emits JSON-lines on stdout. One line per item.
# Each line is a single CompressedObservation-shaped object that
# graph-extract will accept.

emit_memories() {
  curl -fsS --max-time 60 "$URL/agentmemory/memories" \
    | jq -c --arg proj "$PROJECT_PATTERN" '
      .memories
      | map(select(.isLatest != false))
      | map(select($proj == "" or ((.concepts // []) | any(test($proj; "i")))))
      | .[]
      | {
          id: ("obs-graphsweep-mem-" + .id),
          sessionId: (.sessionIds[0] // ("synthetic-mem-" + .id)),
          timestamp: (.createdAt // (now | todate)),
          type: "discovery",
          title: ((.title // .content)[0:80]),
          narrative: .content,
          facts: [],
          concepts: (.concepts // []),
          files: (.files // []),
          importance: 5,
          confidence: 0.85
        }
    '
}

emit_lessons() {
  curl -fsS --max-time 60 "$URL/agentmemory/lessons?limit=10000" \
    | jq -c --arg proj "$PROJECT_PATTERN" '
      .lessons
      | map(select(.deleted != true))
      | map(select($proj == "" or ((.project // "") | test($proj; "i"))))
      | .[]
      | {
          id: ("obs-graphsweep-lsn-" + .id),
          sessionId: ("synthetic-lsn-" + .id),
          timestamp: (.createdAt // (now | todate)),
          type: "discovery",
          title: ((.content // "")[0:80]),
          narrative: .content,
          facts: [],
          concepts: (.tags // []),
          files: [],
          importance: 5,
          confidence: (.confidence // 0.7)
        }
    '
}

emit_summaries() {
  # NOTE: /agentmemory/summaries is not a list endpoint in upstream
  # agentmemory (returns 404). Summaries are KV-stored by sessionId
  # and only the per-session /agentmemory/summaries/<sessionId> shape
  # exists. To enumerate, walk sessions and fetch per-session — same
  # pattern as emit_observations below.
  local session_ids
  session_ids=$(curl -fsS --max-time 60 "$URL/agentmemory/sessions" 2>/dev/null \
    | jq -r --arg proj "$PROJECT_PATTERN" '
      .sessions
      | map(select(.project != null))
      | map(select(($proj == "") or (((.project // "") | split("/") | .[-1]) | test($proj; "i"))))
      | map(select(((.project // "") | test("^(observer|agent-)"; "i")) | not))
      | .[].id
    ' 2>/dev/null) || true
  local sid
  while IFS= read -r sid; do
    [[ -z "$sid" ]] && continue
    # Some upstream builds expose /agentmemory/summary/<id> singular,
    # others /agentmemory/summaries/<id>. Try both; ignore 404s.
    local body
    body=$(curl -fsS --max-time 30 "$URL/agentmemory/summary/$sid" 2>/dev/null) \
      || body=$(curl -fsS --max-time 30 "$URL/agentmemory/summaries/$sid" 2>/dev/null) \
      || continue
    [[ -z "$body" ]] && continue
    jq -c --arg sid "$sid" '
      (.summary // .) as $s
      | select($s != null and ($s | type == "object"))
      | {
          id: ("obs-graphsweep-sum-" + $sid),
          sessionId: $sid,
          timestamp: ($s.createdAt // (now | todate)),
          type: "discovery",
          title: (($s.title // ("session " + $sid))[0:80]),
          narrative: ($s.narrative // ""),
          facts: ($s.keyDecisions // []),
          concepts: ($s.concepts // []),
          files: ($s.filesModified // []),
          importance: 6,
          confidence: 0.9
        }
    ' <<<"$body" 2>/dev/null || true
  done <<<"$session_ids"
}

emit_observations() {
  # Walk every "real" session (skip observer-/agent-/synthetic noise)
  # and pull its observations. Each obs already has the shape graph-extract
  # expects, so passthrough is fine.
  local session_ids
  session_ids=$(curl -fsS --max-time 60 "$URL/agentmemory/sessions" \
    | jq -r --arg proj "$PROJECT_PATTERN" '
      .sessions
      | map(select(.project != null))
      | map(select(($proj == "") or (((.project // "") | split("/") | .[-1]) | test($proj; "i"))))
      | map(select(((.project // "") | test("^(observer|agent-)"; "i")) | not))
      | .[].id
    ')
  local sid count=0
  while IFS= read -r sid; do
    [[ -z "$sid" ]] && continue
    curl -fsS --max-time 60 "$URL/agentmemory/observations?sessionId=$sid" 2>/dev/null \
      | jq -c '(.observations // []) | .[]' 2>/dev/null
    count=$((count+1))
  done <<<"$session_ids"
  echo "[rebuild-graph] emit_observations: walked $count sessions" >&2
}

# ----- batch + POST -----

graph_extract_batch() {
  # $1 = JSON array string of observations
  local payload
  payload=$(jq -nc --argjson obs "$1" '{observations:$obs}')
  curl -sS --connect-timeout 5 --max-time 180 \
    -X POST "$URL/agentmemory/graph/extract" \
    -H 'content-type: application/json' \
    --data "$payload" 2>&1 || echo '{"error":"curl_failed"}'
}

# Reap finished workers from INFLIGHT_PIDS in place. Uses kill -0
# liveness test (bash 3.2 has no `wait -n`).
reap_finished() {
  if [[ ${#INFLIGHT_PIDS[@]} -eq 0 ]]; then
    return 0
  fi
  local survivors=()
  local pid
  for pid in "${INFLIGHT_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      survivors+=("$pid")
    fi
  done
  INFLIGHT_PIDS=()
  if [[ ${#survivors[@]} -gt 0 ]]; then
    INFLIGHT_PIDS=("${survivors[@]}")
  fi
}

# Block until INFLIGHT_PIDS has fewer than $1 live workers. The 0.5s
# back-off is fine when individual LLM calls run for tens of seconds.
wait_inflight_below() {
  local target="$1"
  reap_finished
  while [[ ${#INFLIGHT_PIDS[@]} -ge $target ]]; do
    sleep 0.5
    reap_finished
  done
}

# Dispatch one batch as a background worker. Each worker writes its
# raw response to a per-batch file under $BATCHES_DIR/batch-<N>.json
# which the main loop consumes after the sweep drains.
#
# `trap - EXIT` inside the subshell clears the inherited EXIT trap
# so the worker exiting (after the LLM responds) doesn't trigger the
# parent's restore_env path. Without this, every successful batch
# completion would force-recreate the container while the parent
# was still mid-sweep.
dispatch_batch() {
  local idx="$1"
  local batch_json="$2"
  local out_file="$BATCHES_DIR/batch-${idx}.json"
  ( trap - EXIT; graph_extract_batch "$batch_json" > "$out_file" 2>&1 ) &
  INFLIGHT_PIDS+=("$!")
}

# Read the response file for a batch and emit one progress line.
# Returns 0 on success, 1 on error so the caller can bump counters.
collect_batch() {
  local idx="$1"
  local items="$2"
  local label="$3"   # "" or "(final)" for the trailing batch
  local out_file="$BATCHES_DIR/batch-${idx}.json"
  local resp
  if [[ ! -s "$out_file" ]]; then
    printf '  ERR batch %d %s missing response file\n' "$idx" "$label" >&2
    return 1
  fi
  resp=$(cat "$out_file")
  if jq -e '.error' >/dev/null 2>&1 <<<"$resp"; then
    printf '  ERR batch %d %s: %s\n' "$idx" "$label" \
      "$(jq -r '.error' <<<"$resp" | head -c 120)"
    return 1
  fi
  local na ne
  na=$(jq -r '.nodesAdded // .nodes_added // 0' <<<"$resp" 2>/dev/null)
  ne=$(jq -r '.edgesAdded // .edges_added // 0' <<<"$resp" 2>/dev/null)
  TOTAL_NODES_ADDED=$((TOTAL_NODES_ADDED + ${na:-0}))
  TOTAL_EDGES_ADDED=$((TOTAL_EDGES_ADDED + ${ne:-0}))
  printf '  batch %d (%d items)%s: +%s nodes / +%s edges\n' \
    "$idx" "$items" "${label:+ $label}" "$na" "$ne"
  return 0
}

run_sweep() {
  local type="$1"
  local emit_fn="$2"

  if ! [[ ",$TYPES," == *,$type,* ]]; then
    return 0
  fi

  echo
  echo "=== Sweeping type=$type ==="
  local t0
  t0=$(date +%s)
  local batch_count=0
  local item_count=0
  local errors=0
  local batch_buf="[]"

  # Per-sweep tmpdir for response files. Cleared at the end (or by EXIT).
  BATCHES_DIR=$(mktemp -d -t rebuild-graph-XXXXXXXX)
  # Per-batch item counts so collect_batch can show the right N.
  local -a batch_sizes=()
  TOTAL_NODES_ADDED=0
  TOTAL_EDGES_ADDED=0

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    item_count=$((item_count+1))
    if [[ "$LIMIT" -gt 0 && "$item_count" -gt "$LIMIT" ]]; then
      break
    fi
    batch_buf=$(jq -c --argjson item "$line" '. += [$item]' <<<"$batch_buf")
    local n
    n=$(jq -r 'length' <<<"$batch_buf")
    if (( n >= BATCH_SIZE )); then
      batch_count=$((batch_count+1))
      batch_sizes[$batch_count]=$n
      if [[ "$DRY_RUN" == 1 ]]; then
        printf '  [DRY] batch %d (%d items)\n' "$batch_count" "$n"
      else
        printf '  [dispatch] batch %d (%d items)\n' "$batch_count" "$n"
        wait_inflight_below "$CONCURRENCY"
        dispatch_batch "$batch_count" "$batch_buf"
      fi
      batch_buf="[]"
    fi
  done < <("$emit_fn")

  # Flush trailing partial batch.
  local n
  n=$(jq -r 'length' <<<"$batch_buf")
  if (( n > 0 )); then
    batch_count=$((batch_count+1))
    batch_sizes[$batch_count]=$n
    if [[ "$DRY_RUN" == 1 ]]; then
      printf '  [DRY] batch %d (%d items, final)\n' "$batch_count" "$n"
    else
      printf '  [dispatch] batch %d (%d items, final)\n' "$batch_count" "$n"
      wait_inflight_below "$CONCURRENCY"
      dispatch_batch "$batch_count" "$batch_buf"
    fi
  fi

  # Drain all remaining workers, then aggregate their response files in
  # index order so progress lines stay deterministic.
  if [[ "$DRY_RUN" != 1 ]]; then
    wait_inflight_below 1
    local i
    for (( i=1; i<=batch_count; i++ )); do
      local label=""
      (( i == batch_count && n > 0 && batch_sizes[i] != BATCH_SIZE )) && label="final"
      if ! collect_batch "$i" "${batch_sizes[$i]}" "$label"; then
        errors=$((errors+1))
      fi
    done
  fi

  rm -rf "$BATCHES_DIR"
  BATCHES_DIR=""

  local t1
  t1=$(date +%s)
  printf '  type=%s items=%d batches=%d nodes=+%d edges=+%d errors=%d duration=%ds\n' \
    "$type" "$item_count" "$batch_count" "$TOTAL_NODES_ADDED" "$TOTAL_EDGES_ADDED" \
    "$errors" "$((t1 - t0))"
}

echo "agentmemory graph rebuild — server: $URL"
echo "  types:       $TYPES"
echo "  project:     ${PROJECT_PATTERN:-<all>}"
echo "  batch:       $BATCH_SIZE items per LLM call"
echo "  concurrency: $CONCURRENCY in-flight LLM calls"
[[ "$LIMIT" -gt 0 ]] && echo "  limit:       $LIMIT items per type"
[[ "$DRY_RUN" == 1 ]] && echo "  DRY RUN: no POSTs will be made."
echo

run_sweep memory      emit_memories
run_sweep lesson      emit_lessons
run_sweep summary     emit_summaries
run_sweep observation emit_observations

echo
echo "=== Done. EXIT trap (if set) will restore env. ==="
