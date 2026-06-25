#!/usr/bin/env bash
# Pod entrypoint for the credentialed RunPod path: runs as the RunPod
# dockerStartCmd on a fresh runpod/pytorch image (CUDA + torch preinstalled,
# nothing else). Clones this repo's branch directly from GitHub, installs
# the remaining deps, then runs this pod's shard of run_pipeline.py. All
# secrets arrive as env vars RunPod injects at pod creation
# (scripts/bootstrap_pod.py's `env=` argument) -- never baked into this
# script or git history.
#
# Code distribution is git clone, not the R2 tarball PLAN.md originally
# described (scripts/package_code.py + an R2 download here): this sandbox's
# egress proxy live-confirmed blocks per-account R2 subdomains (TLS
# handshake failure at the proxy, distinct from a normal connection
# refusal), so package_code.py's upload step cannot run from here. GitHub
# is reachable from both this sandbox and the pod, so cloning sidesteps the
# blocker entirely. R2 is untouched for everything else -- the pod still
# uploads clips/manifest/heartbeat to R2 directly using the R2_* env vars
# below, since that traffic never passes through this sandbox's proxy.
#
# UNTESTED: written against the live-verified RunPod REST v1 shape
# (pipeline/runpod_client.py) but never run on an actual pod, since no
# RunPod credentials were available while writing it. Treat the first real
# run as the test; check the pod's logs closely.
set -euo pipefail

GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/rocketsri/podcast.git}"
GIT_BRANCH="${GIT_BRANCH:-claude/podcast-speech-builder-e0kfs5}"

echo "[bootstrap] installing system deps (ffmpeg, git)..."
apt-get update -qq && apt-get install -y -qq ffmpeg git >/dev/null

WORKDIR=/workspace/podcast
clone_url="$GIT_REPO_URL"
if [ -n "${GITHUB_TOKEN:-}" ]; then
    clone_url="https://${GITHUB_TOKEN}@${GIT_REPO_URL#https://}"
fi

echo "[bootstrap] cloning ${GIT_REPO_URL} (branch ${GIT_BRANCH})..."
git clone --depth 1 --branch "$GIT_BRANCH" "$clone_url" "$WORKDIR"
cd "$WORKDIR"

pip install --quiet -r requirements.txt

mkdir -p work
echo "[bootstrap] starting run_pipeline.py (shard=${SHARD:-unset}, pod_id=${POD_ID:-unset})..."
exec python3 run_pipeline.py \
    --db work/pipeline.db \
    --work-dir work \
    --log-path work/pipeline.log \
    --pod-id "${POD_ID:?POD_ID env var required}" \
    --shard "${SHARD:?SHARD env var required}" \
    --device cuda
