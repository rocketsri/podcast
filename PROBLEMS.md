# Problems encountered and how they were fixed

Chronological log of real problems hit while building and running this
pipeline against live data, kept separate from PLAN.md (the forward-looking
design doc) and LIMITATIONS.md (known, accepted weaknesses in the shipped
approach). Entries below are bugs/incidents with a concrete fix.

## 1. R2 reachability was misdiagnosed as a proxy block

**Problem.** Early in the GPU-path work, an attempt to reach R2's
per-account subdomain from the orchestrating sandbox failed with a TLS
handshake error. This was read as "the sandbox's egress proxy blocks R2
per-account subdomains" and used to justify routing code distribution
through `git clone` instead of the originally-planned R2 tarball
(`scripts/package_code.py`).

**Root cause.** The failure was retested and traced to the request being
made against a *placeholder* R2 account id, not a real proxy block. R2 is in
fact reachable from the sandbox.

**Fix.** `infra/bootstrap.sh`'s and `scripts/bootstrap_pod.py`'s docstrings
were corrected to say R2 is reachable from both sides. Code distribution
was *not* switched back to the R2-tarball design, though -- `git clone` was
already working and simpler, so there was no reason to revert once the
real blocker turned out to be a non-issue. R2 continues to be used for
everything the pod itself does (clip/manifest/heartbeat upload).

## 2. RunPod client treated a successful DELETE response as an error

**Problem.** `pipeline/runpod_client.py`'s `_request()` only accepted HTTP
200/201 as success. RunPod's `DELETE /pods/{podId}` (used by
`terminate_pod`) correctly returns **204 No Content** on success, which the
client raised `RunPodError` for.

**Fix.** Added 204 to the accepted status tuple:
`if resp.status_code not in (200, 201, 204): raise RunPodError(...)`.
Found and fixed while launching the first real pod (commit `582b9e4`).

## 3. `create_pod`'s `ports` argument used the wrong type

**Problem.** `RunPodClient.create_pod(ports=...)` was typed and called as a
comma-separated string (`"8080/http"`), but the live RunPod v1 schema
(confirmed against `https://rest.runpod.io/v1/openapi.json`) requires
`ports` to be a JSON array of strings.

**Fix.** Changed the parameter type to `list[str] | None` and the call site
in `scripts/bootstrap_pod.py` to pass `ports=["8080/http"]`. Same commit as
#2 -- both were live-API-only bugs that no amount of reading docs would have
caught without an actual `create_pod` call.

## 4. Missing per-pod episode discovery would have produced a silent empty run

**Problem.** The original GPU-path design (per `PLAN.md`) assumed a single
shared, pre-partitioned SQLite db traveled with the code to every pod, with
each pod processing its assigned `--shard`. But `*.db` is gitignored
(deliberately -- db files don't belong in git) and code reaches each pod via
`git clone`, so no database -- partitioned or otherwise -- ever actually
arrives on a pod. Without a fix, `run_pipeline.py --shard N` would have
started against an empty/nonexistent queue and done nothing, with no error
loud enough to notice quickly.

**Fix.** `infra/bootstrap.sh` now runs `scripts/select_podcasts_free.py` at
boot, before `run_pipeline.py`, so each pod discovers and queues its own
independent batch of real episodes via the free iTunes Search + RSS path.
`run_pipeline.py` is invoked without `--shard` (single-pod mode over
whatever that pod just queued). `scripts/bootstrap_pod.py`'s stale
pre-flight check against the orchestrator's *local* db (a holdover from the
old shared-db design, which would have reported 0 episodes for every shard
and blocked `--confirm` outright) was removed accordingly.

**Tradeoff accepted at the time:** multiple pods running unchanged
`bootstrap.sh` would each discover from the *same* hardcoded query list and
very likely queue heavily overlapping episodes. This was fine for the
single-pod launch it shipped with, but became a real problem once asked to
parallelize across more pods -- see #8.

## 5. CPU/Resemblyzer diarization collapsed multiple real speakers into one cluster

**Problem.** Database inspection of the two completed free-CPU-path
episodes showed one speaker label (`speaker_001`) credited with 97% and 94%
of all clips respectively, despite both shows being clear multi-host
conversations (transcript spot-checks showed `Dave`, `Kevin`, `Paul`, and a
guest all alternating). The diarization had merged distinct speakers into a
single cluster rather than genuinely finding one dominant speaker.

**Root cause.** `pipeline/local_diarize.py`'s clustering step --
`AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
distance_threshold=1.0 - match_threshold)` with `match_threshold=0.75`
(`distance_threshold=0.25`) -- is too permissive for Resemblyzer embeddings
of short (1-8s) same-mic-recording segments. Different real speakers' voice
embeddings ended up within 0.25 cosine distance of each other often enough
to collapse into one cluster.

**Fix.** Rather than re-tuning the CPU path's threshold and re-running it,
the CPU path was retired outright (explicit user instruction: "don't run on
cpus anymore its too slow"). All 3 local CPU worker processes were killed.
The credentialed GPU path's `pipeline/diarize.py` (real pyannote.audio
`speaker-diarization-3.1`, joint segmentation+diarization, overlap-aware,
no hardcoded speaker-count hint) was already the architecturally sound
implementation and needed no model change -- see #9 for the guardrail added
on top of it instead.

**Disposition of the 2 already-completed CPU-path episodes:** their clips
have known-bad diarization and should be excluded from the final deliverable
manifest. They remain useful only as smoke-test evidence that the rest of
the pipeline (ingest -> VAD -> segmentation -> ASR -> quality filtering ->
manifest) runs end-to-end.

## 6. `requirements.txt`'s torch pins likely forced a slow reinstall on every pod boot

**Problem.** The first GPU pod (`podcast-shard-0`) showed no heartbeat in R2
for 25-30+ minutes after creation, well past a normal `pip install -r
requirements.txt` + model-download window.

**Root cause.** `requirements.txt` pinned `torch==2.5.1` /
`torchaudio==2.5.1` exactly, but the RunPod image
(`runpod/pytorch:1.0.7-cu1281-torch271-ubuntu2204`) ships torch **2.7.1**
preinstalled. An exact-version pin forces pip to uninstall the preinstalled
CUDA torch and reinstall a different (CPU or mismatched-CUDA) wheel -- a
multi-GB download -- on every single pod boot, even though the preinstalled
version would have worked fine.

**Fix.** Relaxed the pins to `torch>=2.5.1` / `torchaudio>=2.5.1` in
`requirements.txt`, so pip recognizes the already-installed 2.7.1 satisfies
the constraint and skips reinstalling. Chose `>=` over hardcoding `==2.7.1`
to avoid coupling the pin to one specific image tag. Note: this only helps
pods cloned *after* this fix is pushed -- the already-running shard-0 pod
cloned the old pin and is unaffected by this change.

## 7. Re-running `bootstrap_pod.py` for more pods would collide with the running pod's name

**Problem.** `scripts/bootstrap_pod.py`'s pod-creation loop always started
numbering at `shard_id=0` (`for shard_id in range(args.num_pods): pod_id_label
= f"{prefix}-{shard_id}"`). Running it again with `--num-pods 5` to add more
pods would create `podcast-shard-0` through `podcast-shard-4`, and
`podcast-shard-0` was already the name (and R2 heartbeat key --
`storage.status_key()` is keyed by this exact name) of the pod created in
the first launch. Two pods sharing one name would silently clobber each
other's heartbeat in R2, breaking `poll_status.py`'s monitoring for both.

**Fix.** Added a `--shard-offset` argument to `scripts/bootstrap_pod.py`;
`shard_id` is now computed as `args.shard_offset + i` instead of always
starting at 0. The next launch uses `--shard-offset 1` to start at
`podcast-shard-1`, leaving the running `podcast-shard-0` untouched.

## 8. Launching more pods unchanged would duplicate discovery across all of them

**Problem.** Per #4's accepted tradeoff, `infra/bootstrap.sh` ran a single
hardcoded `--queries` list for every pod. Launching 5 more pods without
changing this would have all 6 pods run the *identical* iTunes Search
queries and very likely discover the same or heavily overlapping episodes
into their separate per-pod databases -- directly defeating the point of
"parallelize across more GPUs" (more *distinct* coverage, not more copies of
the same coverage).

**Fix.** `infra/bootstrap.sh` now branches its `--queries` list on the
`$SHARD` env var (already injected per-pod by `bootstrap_pod.py`'s
`build_env()`) via a `case` statement, giving each of shards 0-5 a distinct,
non-overlapping topic group. Shard 0 keeps the exact query list it was
already running with (so restarting that pod still queues the same content);
shards 1-5 cover disjoint new topics. Any shard index outside 0-5 falls back
to the original generic `select_podcasts_free.py` default list.

## 9. No automated check would have caught problem #5 without manual transcript review

**Problem.** The diarization collapse in #5 was only found by hand: querying
`clips` grouped by `episode_id, speaker_id` and manually reading transcript
samples. Nothing in the pipeline itself would have flagged it -- a pod could
run for hours producing badly-collapsed output with no signal in the
heartbeat or logs.

**Fix (the "better approach" requested).** The real pyannote.audio GPU path
in `pipeline/diarize.py` was already architecturally sound (joint
segmentation+diarization, overlap-aware via `embedding_exclude_overlap`, no
hardcoded speaker-count hint) -- the problem in #5 was specific to the
CPU/Resemblyzer fallback's clustering threshold, not a deficiency in the
GPU model choice. So instead of swapping models, a lightweight, model-agnostic
guardrail was added: `diarize.dominant_speaker_share(turns)` computes the
most-talkative local label's share of total speech time and the number of
distinct labels detected. `pipeline_runner.py`'s
`_ensure_diarized_and_clustered()` calls this right after diarization and,
if `num_labels >= 2` and the dominant share is `>=
clustering.dominant_speaker_warn_threshold` (0.92, new config key in
`config/pipeline.yaml`), logs a warning and records
`run_meta["dominant_speaker_warning_<episode_id>"]` -- queryable later
without re-deriving it from raw clips/transcripts. This is exactly the
signature problem #5 exhibited (96-97% dominance across 2 detected labels)
and would have surfaced it automatically, per-episode, during the run
instead of requiring a manual audit afterward.

## 10. Local disk usage grew unboundedly -- raw/wav/clip files were never deleted after upload

**Problem.** While the 6 pods were running, RunPod's console showed
divergent container-disk usage across pods that had been alive for
comparable wall-clock time (one pod noticeably higher than the rest). This
contradicts `PLAN.md`'s stated design assumption that "per-pod local disk
stays bounded regardless of total corpus size because raw audio is
processed-and-deleted per episode."

**Root cause.** That sentence described the *intended* design but was never
actually implemented. `pipeline_runner.py`'s `_ensure_downloaded()` writes
the raw download to `work/raw/{episode_id}{suffix}`, `_ensure_transcoded()`
writes a 16kHz mono wav to `work/wav/{episode_id}.wav`, and
`_ensure_exported_and_uploaded()` writes every surviving clip to
`work/clips/{podcast_id}/{episode_id}/{clip_id}.flac` -- and nothing in
`pipeline_runner.py`, `ingest.py`, `audio.py`, or `storage.py` ever deleted
any of these three file categories afterward, even once a clip was
confirmed uploaded to R2. Every pod's local disk grows by roughly one raw
file + one transcoded wav + one flac per surviving clip, *per episode,
forever*, against a fixed 30GB `containerDiskInGb` pod quota -- on a 24h
run each pod's `select_podcasts_free.py` queues against the full
150-200-raw-hour `target_corpus` target (not divided across pods, see
problem #8), so a pod that actually got through dozens of episodes would
eventually exhaust its disk and crash outright (not a graceful
checkpoint-and-resume failure -- RunPod doesn't restart a container whose
process exited).

**Fix.** `_ensure_transcoded()` now deletes the raw file immediately after
a successful transcode (confirmed via grep that nothing downstream of that
stage ever reads `local_raw_path` again). `_ensure_exported_and_uploaded()`
now deletes the episode's transcoded wav once the episode reaches "done"
(nothing reads `local_wav_path` past that point either), and removes the
episode's local clip-flac directory too, but only when a real R2
`storage_client` is configured -- in local-only/dev mode (no R2
credentials) the local flac files are the actual deliverable and must
stay. Both deletions sit after the stage that needs the file already
succeeded, so a pod that crashes mid-episode and resumes still finds the
raw/wav file it needs for the stage it's resuming into -- only a *fully
done* episode's intermediate files are freed.

## 11. First episode per pod was picked for hour-target efficiency, not fast feedback

**Problem.** 30+ minutes into the run, R2 still had 0 completed
episodes/clips across all 6 pods, with no way yet to confirm the live GPU
path actually works end-to-end. Looked like it could just be slow, but it
also risked masking a real failure behind a very long first episode.

**Root cause.** `select_podcasts_free.py` selects podcasts
longest-average-duration-first (deliberately, to hit the 150-200-raw-hour
target with fewer podcasts to manage), and `db.list_queued_episodes()`
claimed episodes in `episode_id` (insertion) order -- so a pod's very
first episode was likely to be one of the longest in its whole queue,
which is the worst case both for getting a fast pipeline-works/doesn't-work
signal and for banking any usable hours early in the 24h window.

**Fix.** `list_queued_episodes()` now orders by
`duration_seconds_reported ASC` (NULLs last) instead of `episode_id` --
same selected episode set, but pods claim their shortest known episodes
first. This only takes effect for a queue fetched after this change (i.e.
a fresh `run_pipeline.py` start), so applying it to already-running pods
requires relaunching them.
