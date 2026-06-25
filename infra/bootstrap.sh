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
# described (scripts/package_code.py + an R2 download here) -- kept this way
# even after R2 turned out to be reachable from the orchestrating sandbox
# too (an earlier per-account-subdomain TLS failure was traced to testing
# against a placeholder account id, not an actual proxy block); git clone is
# simpler and was already working, so there was no reason to switch back.
#
# No shared pre-partitioned database arrives with the clone (*.db is
# gitignored, deliberately -- db files don't belong in git). Each pod
# instead runs its own scripts/select_podcasts_free.py at boot to discover
# and queue a fresh, independent batch of real episodes (no PodcastIndex
# credentials configured for this run), then runs run_pipeline.py in
# single-pod mode (no --shard) over whatever it just queued. This means
# multiple concurrent GPU pods would each discover their own separate
# batches rather than splitting one shared pool -- fine for the current
# single-pod launch; true multi-pod sharding would need the discovery step
# centralized and the resulting db handed to each pod (e.g. via R2, now
# that it's confirmed reachable) instead of run independently per pod.
#
# UNTESTED: written against the live-verified RunPod REST v1 shape
# (pipeline/runpod_client.py) but never run on an actual pod before this
# launch. Treat this first real run as the test; check the pod's logs
# closely.
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
echo "[bootstrap] discovering episodes (free iTunes+RSS path, no PodcastIndex creds needed)..."
# Different search terms than run_pipeline_local.py's CPU shards use (see
# scripts/select_podcasts_free.py's DEFAULT_QUERIES) so this pod's GPU time
# goes toward genuinely new episodes instead of re-processing audio the free
# CPU path is already covering concurrently.
python3 scripts/select_podcasts_free.py --db work/pipeline.db \
    --queries "debate" "documentary" "science podcast interview" "history podcast" "panel discussion"

echo "[bootstrap] starting run_pipeline.py (pod_id=${POD_ID:-unset}, shard=${SHARD:-unset})..."
exec python3 run_pipeline.py \
    --db work/pipeline.db \
    --work-dir work \
    --log-path work/pipeline.log \
    --pod-id "${POD_ID:?POD_ID env var required}" \
    --device cuda
