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
# First real run (6 pods) surfaced two bugs that turned a transient failure
# into a permanent, money-burning crash loop: see PROBLEMS.md #12. Both are
# fixed below (idempotent clone, `python3 -m pip`, fail-fast import check).
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

# RunPod restarts this container's entrypoint on a crash (observed directly:
# a pod stuck on the yaml ModuleNotFoundError below was relaunched by RunPod
# every ~16s for 20+ minutes -- contrary to what was previously assumed in
# PROBLEMS.md #10). A bare `git clone` into an already-populated $WORKDIR
# fails outright, which turned that one transient failure into a permanent
# boot loop on every pod, since the second-and-later attempts could never
# even get past the clone step. Make this idempotent: reuse the existing
# clone (fetch + hard reset) if one is already there instead of failing.
if [ -d "$WORKDIR/.git" ]; then
    echo "[bootstrap] ${WORKDIR} already has a clone from an earlier attempt in this container -- updating in place..."
    git -C "$WORKDIR" fetch --depth 1 origin "$GIT_BRANCH"
    git -C "$WORKDIR" checkout -f "$GIT_BRANCH"
    git -C "$WORKDIR" reset --hard "origin/$GIT_BRANCH"
else
    echo "[bootstrap] cloning ${GIT_REPO_URL} (branch ${GIT_BRANCH})..."
    git clone --depth 1 --branch "$GIT_BRANCH" "$clone_url" "$WORKDIR"
fi
cd "$WORKDIR"

# Bare `pip install` resolved to a different interpreter than `python3`
# below it on the actual pod (observed: it returned in under ~15s total --
# far too fast to have really installed pyannote.audio's dependency tree --
# and yaml, requirements.txt's very first package, was then missing from
# the `python3` that runs select_podcasts_free.py). `python3 -m pip`
# guarantees the install target is the same interpreter every later step
# uses. The import check fails loudly and immediately instead of silently,
# so a recurrence shows up as one clear line instead of 20+ minutes of
# pods crash-looping with no GPU/CPU/R2 signal at all.
echo "[bootstrap] installing python deps..."
python3 -m pip install -r requirements.txt
python3 -c "import yaml, torch, pyannote.audio, faster_whisper" \
    || { echo "[bootstrap] FATAL: core deps not importable after install -- aborting" >&2; exit 1; }
echo "[bootstrap] python deps OK"

mkdir -p work
echo "[bootstrap] discovering episodes (free iTunes+RSS path, no PodcastIndex creds needed)..."
# Each GPU pod's $SHARD gets a distinct query group so N pods launched
# together don't all rediscover the same overlapping iTunes Search results
# into separate dbs -- shard 0's list is what the first pod (already
# running before this table existed) launched with, kept as-is here so a
# restart of that pod still queues the same content; shards 1-5 cover
# disjoint topics for --num-pods 5 --shard-offset 1.
case "${SHARD:-0}" in
    0) QUERIES=("debate" "documentary" "science podcast interview" "history podcast" "panel discussion") ;;
    1) QUERIES=("true crime interview" "business podcast interview" "tech podcast interview" "comedy interview podcast" "sports talk show") ;;
    2) QUERIES=("philosophy podcast" "psychology interview" "health podcast interview" "finance podcast interview" "education podcast interview") ;;
    3) QUERIES=("news analysis podcast" "politics interview podcast" "culture podcast interview" "book podcast interview" "film podcast interview") ;;
    4) QUERIES=("startup podcast interview" "music podcast interview" "religion podcast interview" "travel podcast interview" "food podcast interview") ;;
    5) QUERIES=("self improvement podcast" "relationship podcast interview" "parenting podcast interview" "career podcast interview" "leadership podcast interview") ;;
    *) QUERIES=("interview" "conversation" "talk show" "long form interview" "in depth conversation") ;;
esac
python3 scripts/select_podcasts_free.py --db work/pipeline.db --queries "${QUERIES[@]}"

echo "[bootstrap] starting run_pipeline.py (pod_id=${POD_ID:-unset}, shard=${SHARD:-unset})..."
exec python3 run_pipeline.py \
    --db work/pipeline.db \
    --work-dir work \
    --log-path work/pipeline.log \
    --pod-id "${POD_ID:?POD_ID env var required}" \
    --device cuda
