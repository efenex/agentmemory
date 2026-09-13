# Local-all-fixes image: bundles agentmemory's already-built dist/ from
# this checkout (NOT the published npm package) on top of node:22-slim
# with the pinned iii binary from iiidev/iii.
#
# Build context is the agentmemory repo root, not docker/. Run with:
#   docker build -f docker/Dockerfile -t agentmemory-local:latest .
#
# Pattern mirrors deploy/coolify/Dockerfile but installs our own code so
# the running container reflects whatever's currently on local/all-fixes.

ARG III_VERSION=0.11.2

FROM iiidev/iii:${III_VERSION} AS iii-image

FROM node:22-slim

ARG III_VERSION=0.11.2
ARG III_SDK_VERSION=0.11.2

# tini  — proper init so signals reach the agentmemory process
# gosu  — drop root in the entrypoint after fixing /data perms
# curl  — healthcheck
# openssl + ca-certs — TLS to the embedding endpoint (vLLM, etc.)
# procps — ps/top for in-container debugging (CPU/RSS triage)
RUN apt-get update \
 && apt-get install -y --no-install-recommends openssl ca-certificates tini gosu curl procps \
 && rm -rf /var/lib/apt/lists/*

COPY --from=iii-image /app/iii /usr/local/bin/iii

WORKDIR /opt/agentmemory

# Install the production-side npm deps that aren't bundled by tsdown.
# Use a stub package.json so we can pin iii-sdk to the EXACT version
# matching the engine binary above. (Caret ^0.11.2 otherwise resolves
# to 0.11.6 which needs the sandbox-everything worker model agentmemory
# isn't refactored for yet — same constraint as deploy/coolify/Dockerfile.
# `overrides` can't touch direct deps, so we rewrite the version pin
# directly instead.)
COPY package.json ./agentmemory-source-package.json
RUN node -e "const p=require('./agentmemory-source-package.json'); \
    const deps=Object.assign({},p.dependencies||{}); \
    deps['iii-sdk']='${III_SDK_VERSION}'; \
    const stub={name:'agentmemory-runtime',version:'1.0.0',private:true, \
                dependencies:deps}; \
    require('fs').writeFileSync('package.json', JSON.stringify(stub,null,2));" \
 && npm install --omit=dev --omit=optional --no-fund --no-audit --legacy-peer-deps \
 && rm agentmemory-source-package.json

# Our pre-built dist/ (run `npm run build` on the host before docker build).
COPY dist/ ./dist/

# Convenient `agentmemory` command, so `node /opt/agentmemory/dist/cli.mjs`
# doesn't appear everywhere.
RUN ln -s /opt/agentmemory/dist/cli.mjs /usr/local/bin/agentmemory \
 && chmod +x /opt/agentmemory/dist/cli.mjs

ENV AGENTMEMORY_III_VERSION=${III_VERSION} \
    TINI_SUBREAPER=1

EXPOSE 3111 3112 3113 9464 49134

# start-period covers the ~70s iii-engine index load on boot so the
# container reports `starting` (not `unhealthy`) until the engine is up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://127.0.0.1:3111/agentmemory/livez || exit 1

COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/agentmemory-entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/agentmemory-entrypoint.sh"]
