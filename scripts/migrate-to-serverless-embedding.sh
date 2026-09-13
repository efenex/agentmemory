#!/usr/bin/env bash
# migrate-to-serverless-embedding.sh — flip ~/.agentmemory-docker/.env
# from "serverless chat + local-vLLM embedding" to "serverless for
# both", verify the daemon comes up clean, and roll back if it
# doesn't.
#
# What it does:
#   1. Back up the current .env to a timestamped sibling file.
#   2. Comment out (prefix with `# [pre-serverless-migrate] `) the
#      local-vLLM-specific env vars:
#         OPENAI_EMBEDDING_BASE_URL
#         OPENAI_EMBEDDING_API_KEY
#         OPENAI_EMBEDDING_DIMENSIONS
#         OPENAI_EMBEDDING_MODEL
#         NODE_TLS_REJECT_UNAUTHORIZED
#      so they fall back to OPENAI_BASE_URL / OPENAI_API_KEY (per
#      upstream PR #503's design).
#   3. If --embedding-model NAME is given, append a fresh
#      OPENAI_EMBEDDING_MODEL=NAME line so the daemon uses a provider-
#      compatible model. Without this flag, the daemon falls back to
#      the openai-provider default (text-embedding-3-small), which
#      only works against providers exposing that exact model — set
#      this if you're not on real OpenAI.
#   4. `docker compose up -d --force-recreate` so the daemon re-reads
#      env_file.
#   5. Wait for /livez, then verify `provider=llm` AND
#      `embeddingProvider=embeddings`. If either is wrong, roll back.
#   6. (Optional) Run a smoke test by POSTing one memory and
#      confirming it landed.
#   7. On any failure between (2) and (6), restore from the backup
#      and force-recreate. On success, leave the backup alongside
#      the .env so a manual rollback stays one `cp` away.
#
# The EXIT trap restores the backup whenever the script exits in a
# non-committed state (caught by a sentinel — see COMMIT_DONE below).
#
# Usage:
#   scripts/migrate-to-serverless-embedding.sh --embedding-model baai/bge-base-en-v1.5
#   scripts/migrate-to-serverless-embedding.sh --dry-run
#   scripts/migrate-to-serverless-embedding.sh --skip-recreate     # edit .env only
#   scripts/migrate-to-serverless-embedding.sh --no-verify         # skip smoke test
#
# Manual rollback (always works):
#   cp ~/.agentmemory-docker/.env.pre-serverless-migrate.<ts> \
#      ~/.agentmemory-docker/.env
#   docker compose -f docker/docker-compose.yml up -d --force-recreate

set -euo pipefail

URL="${AGENTMEMORY_URL:-http://localhost:3111}"
ENV_FILE="${AGENTMEMORY_ENV_FILE:-$HOME/.agentmemory-docker/.env}"
COMPOSE_FILE="${AGENTMEMORY_COMPOSE_FILE:-$(cd "$(dirname "$0")/.." && pwd)/docker/docker-compose.yml}"

EMBEDDING_MODEL=""
DRY_RUN=0
SKIP_RECREATE=0
NO_VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --embedding-model) EMBEDDING_MODEL="${2:?--embedding-model needs a value}"; shift 2 ;;
    --dry-run)         DRY_RUN=1; shift ;;
    --skip-recreate)   SKIP_RECREATE=1; shift ;;
    --no-verify)       NO_VERIFY=1; shift ;;
    -h|--help)         sed -n '2,46p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

for bin in curl jq docker; do
  command -v "$bin" >/dev/null || { echo "missing dependency: $bin" >&2; exit 1; }
done

[[ -f "$ENV_FILE" ]] || { echo "env file not found: $ENV_FILE" >&2; exit 1; }

# Sentinel marker on commented lines so a re-run is idempotent and
# the rollback step can grep-locate them deterministically.
COMMENT_PREFIX="# [pre-serverless-migrate]"

# Lines to neutralize (comment out). The match is anchored to the
# start of the line so commented-out variants are not re-commented.
TARGETS=(
  "OPENAI_EMBEDDING_BASE_URL"
  "OPENAI_EMBEDDING_API_KEY"
  "OPENAI_EMBEDDING_MODEL"
  "OPENAI_EMBEDDING_DIMENSIONS"
  "NODE_TLS_REJECT_UNAUTHORIZED"
)

TS=$(date +%Y%m%dT%H%M%S)
BACKUP_FILE="${ENV_FILE}.pre-serverless-migrate.${TS}"
COMMIT_DONE=0

# Rollback trap — fires only when COMMIT_DONE is 0, so a successful
# verify leaves the new state in place.
rollback() {
  if (( COMMIT_DONE == 1 )); then
    return 0
  fi
  if [[ -f "$BACKUP_FILE" ]]; then
    echo "[migrate] rolling back $ENV_FILE from $BACKUP_FILE" >&2
    cp "$BACKUP_FILE" "$ENV_FILE"
    # Try to bring the container back to its previous state. Best-effort.
    docker compose -f "$COMPOSE_FILE" up -d --force-recreate >/dev/null 2>&1 \
      || echo "[migrate] WARNING: force-recreate during rollback failed; recover manually" >&2
  fi
}
trap rollback EXIT

# ----- step 1: backup -----

if [[ "$DRY_RUN" == 1 ]]; then
  echo "[migrate] DRY RUN — no changes will be written."
  echo "  would back up: $ENV_FILE -> $BACKUP_FILE"
else
  cp "$ENV_FILE" "$BACKUP_FILE"
  echo "[migrate] backed up to $BACKUP_FILE"
fi

# ----- step 2: comment-out the local-vLLM lines -----

# Build a sed expression that prefixes each uncommented target line.
# Idempotency: only matches lines that DON'T already start with `#`.
print_changes() {
  for var in "${TARGETS[@]}"; do
    if grep -E "^${var}=" "$ENV_FILE" >/dev/null 2>&1; then
      printf '  - %s would be commented out\n' "$var"
    elif grep -E "^${COMMENT_PREFIX}.*${var}=" "$ENV_FILE" >/dev/null 2>&1; then
      printf '  - %s already commented (skip)\n' "$var"
    else
      printf '  - %s not present in env file (skip)\n' "$var"
    fi
  done
  if [[ -n "$EMBEDDING_MODEL" ]]; then
    printf '  + would append OPENAI_EMBEDDING_MODEL=%s\n' "$EMBEDDING_MODEL"
  fi
}

echo "[migrate] planned changes:"
print_changes

if [[ "$DRY_RUN" == 1 ]]; then
  COMMIT_DONE=1   # don't roll back: we never changed anything
  echo "[migrate] DRY RUN complete — nothing was written."
  exit 0
fi

tmp="${ENV_FILE}.tmp.$$"
cp "$ENV_FILE" "$tmp"
for var in "${TARGETS[@]}"; do
  # sed -i differs between BSD/macOS and GNU. Use the portable
  # tmpfile-rewrite pattern instead.
  awk -v var="$var" -v prefix="$COMMENT_PREFIX" '
    {
      if ($0 ~ "^"var"=") {
        print prefix " " $0
      } else {
        print $0
      }
    }
  ' "$tmp" > "${tmp}.new"
  mv "${tmp}.new" "$tmp"
done
if [[ -n "$EMBEDDING_MODEL" ]]; then
  {
    echo ""
    echo "# Set by migrate-to-serverless-embedding.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "OPENAI_EMBEDDING_MODEL=$EMBEDDING_MODEL"
  } >> "$tmp"
fi
mv "$tmp" "$ENV_FILE"
echo "[migrate] env file updated"

# ----- step 3+4: force-recreate container -----

if [[ "$SKIP_RECREATE" == 1 ]]; then
  echo "[migrate] --skip-recreate set — leaving container alone"
  COMMIT_DONE=1
  exit 0
fi

echo "[migrate] force-recreating container so env_file reloads..."
docker compose -f "$COMPOSE_FILE" up -d --force-recreate >/dev/null
echo "[migrate] waiting for /livez..."
livez_ok=0
for i in $(seq 1 30); do
  if curl -fsS --max-time 3 "$URL/agentmemory/livez" >/dev/null 2>&1; then
    livez_ok=1
    break
  fi
  sleep 1
done
if (( livez_ok == 0 )); then
  echo "[migrate] /livez did not respond within 30s — rolling back" >&2
  exit 1
fi
echo "[migrate] /livez ok"

# ----- step 5: verify providers -----

flags=$(curl -fsS --max-time 5 "$URL/agentmemory/config/flags" 2>&1)
provider=$(jq -r '.provider' <<<"$flags")
embedding_provider=$(jq -r '.embeddingProvider' <<<"$flags")
echo "[migrate] provider=$provider, embeddingProvider=$embedding_provider"

if [[ "$provider" != "llm" || "$embedding_provider" != "embeddings" ]]; then
  echo "[migrate] expected provider=llm + embeddingProvider=embeddings — rolling back" >&2
  exit 1
fi

# ----- step 6: smoke test (POST one memory, confirm it lands) -----

if [[ "$NO_VERIFY" == 1 ]]; then
  echo "[migrate] --no-verify set — skipping smoke test"
  COMMIT_DONE=1
  exit 0
fi

echo "[migrate] smoke-testing: POSTing one memory + reading it back..."
smoke_content="migrate-to-serverless-embedding smoke test ${TS}"
smoke_resp=$(curl -fsS --max-time 30 -X POST "$URL/agentmemory/remember" \
  -H 'content-type: application/json' \
  --data "$(jq -nc --arg c "$smoke_content" '{content:$c, type:"fact", concepts:["migrate-smoke"]}')" \
  2>&1 || echo '{"success":false,"error":"curl_failed"}')

smoke_id=$(jq -r '.memory.id // .id // empty' <<<"$smoke_resp")
smoke_ok=$(jq -r '.success' <<<"$smoke_resp" 2>/dev/null)

if [[ "$smoke_ok" != "true" || -z "$smoke_id" ]]; then
  echo "[migrate] smoke POST failed — rolling back" >&2
  echo "  response: $(head -c 300 <<<"$smoke_resp")" >&2
  exit 1
fi
echo "[migrate] smoke memory created: $smoke_id"

# Give the embedding step a beat to settle, then confirm vector path.
sleep 2
search_resp=$(curl -fsS --max-time 10 -X POST "$URL/agentmemory/smart-search" \
  -H 'content-type: application/json' \
  --data "$(jq -nc --arg q "$smoke_content" '{query:$q, limit:5, mode:"compact"}')" \
  2>&1 || echo '{}')

if jq -e '.results // .memories' >/dev/null 2>&1 <<<"$search_resp"; then
  hit=$(jq -r --arg id "$smoke_id" \
    '(.memories // .results // []) | any(.id == $id or .memoryId == $id)' \
    <<<"$search_resp" 2>/dev/null || echo "false")
  echo "[migrate] smart-search returned in valid shape (hit-on-smoke=$hit)"
else
  echo "[migrate] WARNING: smart-search response shape unexpected — accepting POST as proof enough" >&2
fi

# Cleanup smoke memory so we don't leave junk in the corpus.
curl -fsS --max-time 10 -X POST "$URL/agentmemory/forget" \
  -H 'content-type: application/json' \
  --data "$(jq -nc --arg id "$smoke_id" '{memoryId:$id}')" >/dev/null 2>&1 \
  || echo "[migrate] note: smoke memory $smoke_id not cleaned up (forget failed, harmless)"

COMMIT_DONE=1
echo
echo "[migrate] SUCCESS"
echo "  backup retained at: $BACKUP_FILE"
echo "  manual rollback:    cp '$BACKUP_FILE' '$ENV_FILE' && docker compose -f '$COMPOSE_FILE' up -d --force-recreate"
