# Local-all-fixes image: builds agentmemory's dist/ from this checkout (NOT
# the published npm package) and bundles it on top of node:22-slim with the
# pinned iii binary from iiidev/iii.
#
# Build context is the agentmemory repo root, not docker/. Run with:
#   docker build -f docker/Dockerfile -t agentmemory-local:latest .
#
# Multi-stage, so a host needs only git + docker (no node): a deploy can clone
# a pinned ref and build it as-is. Base images are pinned by multi-arch index
# digest (valid on amd64 and arm64); bump tag and digest together.
#
# Pattern mirrors deploy/coolify/Dockerfile but installs our own code so
# the running container reflects the checked-out ref.

ARG III_VERSION=0.11.2
ARG NODE_IMAGE=node:22-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5
ARG III_IMAGE=iiidev/iii:0.11.2@sha256:15f8d4ed16c0bec350b98f4e18ed04498b1fc5ccc50585e087b736717300cf26

FROM ${NODE_IMAGE} AS build
WORKDIR /src
COPY package.json tsconfig.json tsdown.config.ts iii-config.yaml iii-config.docker.yaml docker-compose.yml ./
COPY src/ ./src/
RUN npm install --no-fund --no-audit --legacy-peer-deps && npm run build

FROM ${III_IMAGE} AS iii-image

FROM ${NODE_IMAGE}

ARG III_VERSION=0.11.2

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

# Install the production-side npm deps that aren't bundled by tsdown, from a
# stub package.json. The repo commits no lockfile, so the direct deps are
# pinned to exact versions from docker/runtime-deps.json (the set verified on
# the live deployment); a dependency missing from that file fails the build
# rather than silently floating. iii-sdk must match the engine binary above
# (^0.11.2 would resolve to 0.11.6, which needs the sandbox-everything worker
# model agentmemory isn't refactored for yet — same constraint as
# deploy/coolify/Dockerfile). Transitive deps still resolve at build time.
COPY package.json ./agentmemory-source-package.json
COPY docker/runtime-deps.json ./runtime-deps.json
RUN node -e "const p=require('./agentmemory-source-package.json'); \
    const pins=require('./runtime-deps.json'); \
    const missing=Object.keys(p.dependencies||{}).filter(d=>!(d in pins)); \
    if(missing.length){console.error('docker/runtime-deps.json lacks: '+missing.join(', '));process.exit(1);} \
    const deps={}; for(const d of Object.keys(p.dependencies)) deps[d]=pins[d]; \
    const stub={name:'agentmemory-runtime',version:'1.0.0',private:true, \
                dependencies:deps}; \
    require('fs').writeFileSync('package.json', JSON.stringify(stub,null,2));" \
 && npm install --omit=dev --omit=optional --no-fund --no-audit --legacy-peer-deps \
 && rm agentmemory-source-package.json runtime-deps.json

COPY --from=build /src/dist/ ./dist/

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
