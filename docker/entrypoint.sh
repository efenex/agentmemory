#!/bin/sh
# agentmemory container entrypoint.
#
# Runs as root long enough to:
#   1. Ensure /data exists and is owned by `node:node`. Bind-mounts on
#      macOS Docker Desktop usually translate UIDs transparently, but
#      named volumes start root-owned 755 and need an explicit chown.
#   2. Overwrite the dist/iii-config.yaml with one that binds 0.0.0.0
#      and uses absolute /data paths. The shipped dist/iii-config.yaml
#      binds 127.0.0.1 (correct for native macOS) — that won't accept
#      connections from outside the container.
#
# Then execs the agentmemory CLI under `gosu node:node`. The CLI sees
# /usr/local/bin/iii on PATH and starts the engine "native" (no
# docker-in-docker recursion).

set -eu

DATA_DIR="${AGENTMEMORY_DATA_DIR:-/data}"
III_CONFIG="/opt/agentmemory/dist/iii-config.yaml"
# The CLI's first-run onboarding wizard reads from stdin and blocks the
# whole process; pre-seed its preferences.json so the daemon skips it.
# Lives at ~/.agentmemory/preferences.json from the node user's POV.
NODE_HOME="/home/node"
PREFS_DIR="$NODE_HOME/.agentmemory"
PREFS_FILE="$PREFS_DIR/preferences.json"

mkdir -p "$DATA_DIR" "$PREFS_DIR"
chown -R node:node "$DATA_DIR" "$NODE_HOME"

if [ ! -f "$PREFS_FILE" ]; then
  NOW="$(date -u +%Y-%m-%dT%H:%M:%S.000Z)"
  cat > "$PREFS_FILE" <<EOF
{
  "firstRunAt": "$NOW",
  "skipSplash": true,
  "lastSeenVersion": "container",
  "selectedAgents": [],
  "selectedProvider": "openai",
  "onboardingCompletedAt": "$NOW"
}
EOF
  chown node:node "$PREFS_FILE"
fi

cat > "$III_CONFIG" <<EOF
workers:
  - name: iii-http
    config:
      port: 3111
      host: 0.0.0.0
      default_timeout: 180000
      cors:
        # Inline, not a block list: the launcher (src/cli/engine-config.ts,
        # setManagedCorsOrigins) rewrites this one line; a block list left
        # its items dangling under the new line and iii refused the YAML
        # (seen on a 2026-09-13 cutover).
        allowed_origins: ["http://localhost:3111", "http://localhost:3113", "http://127.0.0.1:3111", "http://127.0.0.1:3113"]
        allowed_methods: [GET, POST, PUT, DELETE, OPTIONS]
  - name: iii-state
    config:
      adapter:
        name: kv
        config:
          store_method: file_based
          file_path: ${DATA_DIR}/state_store.db
  - name: iii-queue
    config:
      adapter:
        name: builtin
  - name: iii-pubsub
    config:
      adapter:
        name: local
  - name: iii-cron
    config:
      adapter:
        name: kv
  - name: iii-stream
    config:
      port: 3112
      host: 0.0.0.0
      adapter:
        name: kv
        config:
          store_method: file_based
          file_path: ${DATA_DIR}/stream_store
  - name: iii-observability
    config:
      enabled: true
      service_name: agentmemory
      exporter: memory
      sampling_ratio: 1.0
      metrics_enabled: true
      logs_enabled: true
      logs_console_output: true
EOF
chown node:node "$III_CONFIG"

# `agentmemory` (no args) is the right invocation: main() starts the
# detached iii-engine, imports the server module (which holds the
# event loop open via its sockets), and the Node process stays alive
# until killed. Pass-through any args the user added to CMD.
exec gosu node:node /usr/local/bin/agentmemory "$@"
