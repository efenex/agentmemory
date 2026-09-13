#!/usr/bin/env bash
# Drive /agentmemory/backfill/vectors-missing in a loop until embedded=0.
# Each call processes up to BACKFILL_CHUNK observations, returns HTTP 200
# with a progress summary, and we re-invoke.
#
# Usage:
#   AGENTMEMORY_URL=http://127.0.0.1:3111 scripts/backfill-embeddings.sh
# Env:
#   BACKFILL_CHUNK=2000  — embed budget per call (default 2000 ≈ 4-5 min at 7/sec)
#   SLEEP_BETWEEN=2      — seconds between iterations (let vLLM breathe)
#   CALL_TIMEOUT=3600    — per-call curl timeout. Must exceed
#                          BACKFILL_CHUNK / embed-rate, INCLUDING the
#                          corpus walk each call does to find the missing
#                          ones. A timeout is not fatal (embeddings are
#                          persisted and the next call skips them) but it
#                          wastes a whole walk, so size it generously —
#                          a fast local GPU makes large chunks attractive
#                          and large chunks are exactly what blew the old
#                          hard-coded 600s.

set -euo pipefail

AM_URL="${AGENTMEMORY_URL:-http://127.0.0.1:3111}"
CHUNK="${BACKFILL_CHUNK:-2000}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-2}"
CALL_TIMEOUT="${CALL_TIMEOUT:-3600}"

i=0
total_embedded=0
started=$(date +%s)
while true; do
  i=$((i + 1))
  echo "[$(date +%H:%M:%S)] iter $i — POST /backfill/vectors-missing {maxToEmbed: $CHUNK}"
  resp=$(curl -sS --max-time "$CALL_TIMEOUT" -X POST "$AM_URL/agentmemory/backfill/vectors-missing" \
    -H "Content-Type: application/json" -d "{\"maxToEmbed\": $CHUNK}" 2>&1) || {
    echo "  curl failed: $resp"
    sleep 5
    continue
  }
  embedded=$(echo "$resp" | jq -r '.embedded // 0')
  checked=$(echo "$resp" | jq -r '.checked // 0')
  skipped=$(echo "$resp" | jq -r '.skipped // 0')
  capped=$(echo "$resp" | jq -r '.capped // false')
  total_embedded=$((total_embedded + embedded))
  elapsed=$(( $(date +%s) - started ))
  rate="0"
  if [ "$elapsed" -gt 0 ]; then
    rate=$(awk "BEGIN { printf \"%.1f\", $total_embedded / $elapsed }")
  fi
  echo "  embedded=$embedded checked=$checked skipped=$skipped capped=$capped  | total=$total_embedded in ${elapsed}s (${rate}/s)"
  if [ "$embedded" = "0" ]; then
    echo ""
    echo "Done. No missing vectors remain. Total embedded: $total_embedded"
    exit 0
  fi
  sleep "$SLEEP_BETWEEN"
done
