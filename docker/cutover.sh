#!/usr/bin/env bash
# Migrate agentmemory from the host /opt/homebrew install to the
# container. Idempotent: each step checks current state before acting.
#
# What it does (in order):
#   1. Confirm dist/ is built and the image is in docker.
#   2. Create ~/.agentmemory-docker/ if missing.
#   3. Copy ~/.agentmemory/.env  → ~/.agentmemory-docker/.env   (if absent)
#   4. Copy host state_store.db  → ~/.agentmemory-docker/       (if absent)
#   5. Stop the host daemon (pkill -9 the iii + node processes).
#   6. Stop any side-port smoke container that's still up.
#   7. docker compose up -d (canonical ports).
#   8. Wait for /livez and re-check provider detection.
#
# Run from the agentmemory repo root:
#   bash docker/cutover.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

HOST_DATA_DIR="$HOME/.agentmemory"
CONTAINER_DATA_DIR="$HOME/.agentmemory-docker"
# The daemon currently writes its state store next to the agentmemory
# repo (per the project handoff memory). Adjust this if your host install
# uses a different working directory.
HOST_STATE_STORE="$REPO_ROOT/data/state_store.db"
HOST_STREAM_STORE="$REPO_ROOT/data/stream_store"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '   \033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[1;33m!\033[0m %s\n' "$*" >&2; }

step "Verify build artifacts present"
[[ -f dist/cli.mjs ]] || { warn "dist/cli.mjs missing — run 'npm run build' first"; exit 1; }
ok "dist/cli.mjs present"

# Order matters: docker compose evaluates env_file: at every invocation
# (build, up, ps...). The file must exist before any docker compose call
# against docker-compose.yml, so do data-dir setup first.

step "Prepare ~/.agentmemory-docker/"
mkdir -p "$CONTAINER_DATA_DIR"
ok "$CONTAINER_DATA_DIR"

step "Copy .env into container data dir"
if [[ -f "$CONTAINER_DATA_DIR/.env" ]]; then
  ok ".env already present (not overwriting)"
elif [[ -f "$HOST_DATA_DIR/.env" ]]; then
  cp "$HOST_DATA_DIR/.env" "$CONTAINER_DATA_DIR/.env"
  ok "copied from $HOST_DATA_DIR/.env"
else
  warn "no .env found at $HOST_DATA_DIR/.env — creating empty stub so compose doesn't fail"
  : > "$CONTAINER_DATA_DIR/.env"
fi

if ! docker image inspect agentmemory-local:local-all-fixes >/dev/null 2>&1; then
  step "Build container image (first time)"
  docker compose -f docker/docker-compose.yml build
fi
ok "image agentmemory-local:local-all-fixes present"

step "Copy state store into container data dir"
if [[ -e "$CONTAINER_DATA_DIR/state_store.db" ]]; then
  ok "state_store.db already present (not overwriting)"
elif [[ -e "$HOST_STATE_STORE" ]]; then
  cp -R "$HOST_STATE_STORE" "$CONTAINER_DATA_DIR/"
  ok "copied $HOST_STATE_STORE"
else
  warn "no state_store at $HOST_STATE_STORE — container starts with empty corpus"
fi

if [[ -e "$HOST_STREAM_STORE" && ! -e "$CONTAINER_DATA_DIR/stream_store" ]]; then
  cp -R "$HOST_STREAM_STORE" "$CONTAINER_DATA_DIR/"
  ok "copied stream_store"
fi

step "Stop host /opt/homebrew daemon"
if pgrep -f "iii|node.*agentmemory|node /opt/homebrew" >/dev/null; then
  pkill -9 -f "iii|node.*agentmemory|node /opt/homebrew" || true
  sleep 1
  ok "host daemon killed"
else
  ok "no host daemon running"
fi

step "Tear down smoke container if present"
if docker ps -a --format '{{.Names}}' | grep -q "^agentmemory-smoke$"; then
  docker compose -f docker/docker-compose.smoke.yml down >/dev/null 2>&1 || true
  ok "smoke container removed"
fi

step "Bring up cutover container"
# Defensive down: a previous failed `up` (e.g. port already-in-use) can
# leave a Created container that reserves all our published ports, and
# the next `up` then errors "address already in use" against itself.
# Safe no-op when nothing's there.
docker compose -f docker/docker-compose.yml down --remove-orphans >/dev/null 2>&1 || true
docker compose -f docker/docker-compose.yml up -d
ok "container started"

step "Wait for /livez (up to 30s)"
for i in $(seq 1 30); do
  if curl -fsS http://localhost:3111/agentmemory/livez >/dev/null 2>&1; then
    ok "livez ok after ${i}s"
    break
  fi
  sleep 1
done

step "Confirm provider detection"
curl -fsS http://localhost:3111/agentmemory/config/flags 2>/dev/null \
  | jq '{provider, embeddingProvider, version}' || warn "config/flags fetch failed"

step "Run diagnostics"
curl -fsS http://localhost:3111/agentmemory/diagnostics -X POST -H 'content-type: application/json' -d '{}' 2>/dev/null \
  | jq '.summary' || warn "diagnostics fetch failed"

printf '\n\033[1;32mDone.\033[0m Viewer: http://localhost:3113\n'
printf 'Logs: docker compose -f docker/docker-compose.yml logs -f\n'
printf 'Roll back: docker compose -f docker/docker-compose.yml down && start the /opt/homebrew daemon again\n'
