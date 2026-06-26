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

## 12. All 6 relaunched pods crash-looped for 20+ minutes, burning money with zero output

**Problem.** After the #10/#11 relaunch, all 6 pods sat at 0% GPU, 0% CPU,
and zero R2 objects for 20-35 minutes -- looked at first like a slow but
healthy boot (heavy `pyannote.audio` pip install, model download,
discovery). Pulling `podcast-shard-5`'s actual container logs (RunPod
GraphQL/REST expose no log endpoint we have access to, so this required
the user to paste them in manually) showed something much worse: the pod
was not slowly booting, it was crash-restarting every ~16 seconds, in a
loop with no exit condition, the entire time.

**Root cause, two compounding bugs.** (1) `infra/bootstrap.sh` ran bare
`pip install --quiet -r requirements.txt`, which returned in well under
15 seconds total (clone + cd + install + mkdir combined) -- far too fast
to have actually installed `pyannote.audio`'s real dependency tree. It
resolved to a different interpreter/environment than the `python3` used
to run `select_podcasts_free.py` immediately after, so `import yaml`
(requirements.txt's first package) raised `ModuleNotFoundError` and the
script exited nonzero under `set -euo pipefail`. (2) This is itself just
one crash -- the part that made it unrecoverable was that RunPod restarts
this container's entrypoint automatically on exit (directly observed:
~16s between identical log cycles, for 20+ minutes straight). That
contradicts what problem #10 assumed ("RunPod doesn't restart a container
whose process exited") -- it does, at least for this pod/image
configuration. Once bug #1 crashed the container the first time, every
subsequent restart's `git clone` failed immediately too
(`destination path '/workspace/podcast' already exists and is not an
empty directory`, since the leftover directory from the first attempt was
never cleaned up), permanently locking the pod into a loop that could
never get past the clone step again -- regardless of whether the original
pip/yaml bug would have eventually self-resolved.

**Fix.** `infra/bootstrap.sh`: (a) made the clone step idempotent --
if `$WORKDIR/.git` already exists (an earlier attempt in the same
container), `git fetch` + `reset --hard` in place instead of failing;
(b) replaced the bare `pip install` with `python3 -m pip install` so
the install target is guaranteed to be the same interpreter every later
step uses; (c) added a fail-fast `python3 -c "import yaml, torch,
pyannote.audio, faster_whisper"` check immediately after install that
exits with a clear `[bootstrap] FATAL` message instead of letting a
missing dependency surface 20+ lines deep in `select_podcasts_free.py`'s
traceback. Problem #10's "RunPod doesn't restart a container whose
process exited" is left as-is below (historical record of what was
believed at the time), but this entry supersedes it: assume RunPod
**will** retry a crashed container's entrypoint, which makes idempotent
boot steps a correctness requirement, not a nice-to-have.

**Cost impact.** All 6 pods ran in this useless loop for the entire
20-35 minute window between the #10/#11 relaunch and this fix --
6 pods x ~0.5h x ~$0.30/hr is small in absolute dollars, but 100% of it
produced zero usable output, and the same bug would have silently
repeated on every future relaunch until logs were manually pulled and
read by a human, since no telemetry channel available to this session
(GraphQL runtime stats, R2 heartbeat) can distinguish a crash loop from a
slow-but-healthy boot.

## 13. One pod landed on a RunPod host with an incompatible GPU driver -- a hardware issue, not a code bug

**Problem.** After the #12 fix was deployed to all 6 relaunched pods,
direct RunPod GraphQL polling showed `podcast-shard-5` flat at exactly 0%
CPU and 0% GPU across multiple polls 30+ seconds apart, with zero monitor
notifications since launch -- while shards 0-4 all showed at least brief
CPU activity in the same window. The user independently corroborated this
from the RunPod console: shard-5 showed 0% disk used, while shards 0-4
were at 29-36%. This looked at first like it could be a recurrence of
#12, so the pod was restarted (RunPod REST `POST /pods/{id}/restart`) as
a low-cost first attempt.

**Root cause.** The restart did not help, and the user pulled shard-5's
actual container logs, which showed the real cause: the container never
started at all, at the Docker/OCI level, before `infra/bootstrap.sh` (or
any application code) ever ran:
```
error starting container: ... OCI runtime create failed: runc create
failed: ... nvidia-container-cli: requirement error: unsatisfied
condition: cuda>=12.8, please update your driver to a newer version, or
use an earlier cuda container: unknown
```
The `runpod/pytorch:1.0.7-cu1281-torch271-ubuntu2204` image requires
CUDA >= 12.8, but the specific physical host this pod was scheduled onto
(RunPod Community Cloud, `machineId x3fo8pyehccc`) had an older NVIDIA
driver that doesn't satisfy that. Docker retried container creation
every ~16 seconds, continuously, from pod creation through at least 13+
minutes later with zero self-recovery -- explaining the flat 0%
CPU/GPU/disk: there was never a running process to measure. This is a
**host/hardware incompatibility**, entirely outside this repo's code --
no edit to `infra/bootstrap.sh`, `requirements.txt`, or anything else
in-repo could have prevented or fixed it. It is also why the restart
didn't help: restarting a pod via the RunPod API keeps it pinned to the
same physical machine, and the machine's driver was the broken part.

**Fix.** Terminated the affected pod and created a brand new one in its
place with the same shard config (`bootstrap_pod.py --num-pods 1
--shard-offset 5 --confirm`) so RunPod would schedule it onto a
different host. Confirmed via GraphQL (`machineId`) that the new pod
landed on a different machine than the broken one before considering it
resolved.

**Takeaway.** Community Cloud hosts are operator-owned and can have
stale drivers; a single bad host is a real, if infrequent, possibility
and looks identical from the outside to a slow-but-healthy boot (zero
CPU/GPU/disk activity, no logs reaching any telemetry channel this
session has API access to) until real container logs are pulled. The
diagnostic signature that distinguished it from #12: *zero* lines ever
appear in the container's logs at all (not even the first `[bootstrap]`
echo), because the failure is at container creation, a layer below
where `bootstrap.sh` runs.

## 14. Relaxed torch/torchaudio pins let pip drift to versions pyannote.audio can't import

**Problem.** After the #13 host-migration fix, shards 0-4 (all relaunched
with the #12 fix) looked idle from telemetry alone -- near-zero/negative
`uptimeInSeconds`, 0% GPU, no R2 heartbeats -- but the user flagged that
the RunPod console showed them visibly restarting on a loop. The Docker
engine-event log (system log) for shard-4 showed a clean image pull, one
successful container start, ~2.5 minutes of activity, then an unbroken
`start container ... begin` loop every ~16s for 10.5+ minutes with **no
error line at all** -- a third pattern, distinct from both #12 (app bug,
full traceback visible in that same log) and #13 (host driver, explicit
`nvidia-container-cli` error every cycle). The engine-event log doesn't
carry the container's own stdout, so the actual cause was invisible until
the user pulled the separate **application/container log** (a different
tab in the RunPod console).

**Root cause.** That log showed `bootstrap.sh` itself failing, every
cycle, at its own fail-fast guard from the #12 fix:
```
File ".../pyannote/audio/core/io.py", line 60, in <module>
    ) -> torchaudio.AudioMetaData:
AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'
[bootstrap] FATAL: core deps not importable after install -- aborting
```
`requirements.txt` pinned `torch>=2.5.1` / `torchaudio>=2.5.1` (relaxed
from exact `==2.5.1` pins in the #13-adjacent multi-pod-scale-out commit,
specifically so pip would treat the RunPod image's preinstalled torch
2.7.1 as already-satisfied and skip a slow reinstall). In practice pip's
resolver did not stop at the preinstalled version -- the install log shows
it pulled `torch-2.12.1` and `torchaudio-2.11.0`, both far newer than
`pyannote.audio==3.3.2` was built against. `pyannote.audio` declares no
upper bound on torch/torchaudio in its own package metadata, so pip
happily resolves the combination; the incompatibility only surfaces at
*import time*, as an `AttributeError` on an API pyannote's code still
expects. Because `bootstrap.sh` reinstalls `requirements.txt`
unconditionally on every restart (not just once per container), every
single restart hit the same resolution and the same import failure,
forever -- explaining the all-zero telemetry (the failure is in
`bootstrap.sh`, well before `run_pipeline.py` and its status server ever
start) with no Docker-level error (the container itself starts and runs
fine; it's the application script that deliberately `exit 1`s).

**Fix.** Reverted `requirements.txt` to the exact pins proven to work
before the relaxation: `torch==2.5.1` / `torchaudio==2.5.1`. No pod
restart/terminate was needed -- `bootstrap.sh` already re-clones the
branch HEAD on every restart, so the already-looping pods pick up the fix
on their next automatic cycle.

**Takeaway.** An unbounded `>=` pin on a fast-moving package (torch) is
not equivalent to "prefer the preinstalled version" -- pip's resolver can
and did drift to the newest release on PyPI instead, several minor
versions past what a pinned, less-actively-maintained dependency
(`pyannote.audio==3.3.2`) was validated against. A correctness-critical
transitive dependency like this needs an explicit pin (or at least an
upper bound), not just a floor; the "skip a slow reinstall" optimization
that motivated the floor-only pin should have been scoped with an upper
bound from the start. The diagnostic signature that distinguished this
from #12 and #13: the Docker/system log alone showed a clean restart loop
with *no* error text anywhere -- only the separate application-log tab
revealed the actual Python traceback and the deliberate `bootstrap.sh`
abort.

## 15. Same class of bug, one dependency deeper: unpinned huggingface_hub broke pyannote's own hf_hub_download call

**Problem.** Right after the #14 fix shipped, shard-4's restart loop
changed shape but didn't stop: `bootstrap.sh` now passed its import check
("`[bootstrap] python deps OK`") and `run_pipeline.py` actually started
for the first time -- progress -- but it then crashed within ~2 seconds,
every cycle:
```
File ".../pyannote/audio/core/pipeline.py", line 90, in from_pretrained
    config_yml = hf_hub_download(...)
File ".../huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
TypeError: hf_hub_download() got an unexpected keyword argument 'use_auth_token'
```

**Root cause.** Identical mechanism to #14, one level further down the
dependency tree: `requirements.txt` never pinned `huggingface_hub` at
all, so pip resolved it to the newest release (`1.21.0`). `pyannote.audio`
3.3.2's own internal code (`pipeline.py:90`) still calls
`hf_hub_download(..., use_auth_token=hf_token)` -- an argument name
`huggingface_hub` has since removed in favor of `token`. Pinning the
direct dependencies (`torch`/`torchaudio`) wasn't enough; any transitive
dependency pyannote.audio touches at runtime without us pinning it is
just as free to drift to an incompatible newer release.

**Fix.** Added an explicit `huggingface_hub==0.25.2` pin to
`requirements.txt` (the last release before the `use_auth_token` removal).
Verified locally first, without needing a GPU: installed
`huggingface_hub==0.25.2` alone in a throwaway venv and called
`hf_hub_download(..., use_auth_token=...)` against a fake repo -- it
raised a normal `RepositoryNotFoundError`, not a `TypeError`, confirming
the kwarg is still accepted. Then ran `pip install --dry-run -r
requirements.txt` (with the new pin appended) and confirmed it resolves
clean with no version conflicts and without disturbing the `torch==2.5.1`
/ `torchaudio==2.5.1` pins from #14. No pod action needed, same reasoning
as #14: the next automatic restart re-clones the branch and picks up the
new pin.

**Takeaway.** A library pinned to an exact version (`pyannote.audio==
3.3.2`) is only as stable as *its own* unpinned dependencies -- pinning
the direct, obviously-relevant package (torch) doesn't protect against
every other transitive dependency it calls into at runtime. Worth
auditing the rest of pyannote.audio's dependency tree the same way before
assuming this class of bug is fully closed.

## 16. Gated HF model access was never actually granted -- and smoke_test.py's own check gave a false "confirmed"

**Problem.** Acting on "look for all possible problems like this and fix
them," audited the rest of the model-loading chain locally (CPU venv,
exact pinned `requirements.txt`, actually *calling* `vad.load_model()`,
`diarize.load_pipeline()`, and `asr.load_model()` rather than just
importing). `vad` and `asr` (faster-whisper/ctranslate2/tokenizers) both
loaded and ran clean -- no further version-drift bugs there. `diarize`
failed, but not with a version error:
```
GatedRepoError: 403 Client Error ... Cannot access gated repo for url
https://huggingface.co/pyannote/speaker-diarization-3.1/resolve/main/config.yaml.
Access to model pyannote/speaker-diarization-3.1 is restricted and you
are not in the authorized list.
```
The same failure reproduces for `pyannote/segmentation-3.0`, the gated
sub-model `Pipeline.from_pretrained()` loads internally for the
segmentation step. Both confirmed with the exact `HF_TOKEN` from `.env`
-- the same token every pod uses -- so every one of the 6 shards is
blocked here regardless of the #14/#15 fixes.

Worse: `scripts/smoke_test.py`'s `check_huggingface()` had already been
run earlier in the project and reported `[OK] huggingface: token valid
(user=rocketsri), gated model access confirmed` -- a false positive that
likely gave false confidence before the pods were ever launched.

**Root cause.** `check_huggingface()` validated gated access with
`HfApi.model_info(GATED_MODEL_ID, token=...)`. `model_info()` only reads
repo *metadata* (it returns fine even with `gated: "auto"` info) and does
not enforce the per-user gate -- confirmed directly: `model_info()`
succeeds with this exact token, but `hf_hub_download()` of an actual file
from the same repo, with the same token, 403s. The gate is only enforced
on real file fetches, which is exactly what `Pipeline.from_pretrained()`
needs and what the smoke test should have exercised instead of a
metadata-only call.

**Fix.** Changed `check_huggingface()` to call `hf_hub_download(repo_id,
"config.yaml", token=...)` against both gated repos actually used
(`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`)
instead of `model_info()`. Re-ran the smoke test: it now correctly
reports `[FAIL] huggingface: token valid (user=rocketsri) but gated model
access failed for pyannote/speaker-diarization-3.1 -- accept the license
at https://huggingface.co/pyannote/speaker-diarization-3.1`.

This part requires action from whoever holds the `rocketsri` HF account
tied to `HF_TOKEN` -- visit and accept the user-conditions agreement
(logged in as that account) on:
  - https://huggingface.co/pyannote/speaker-diarization-3.1
  - https://huggingface.co/pyannote/segmentation-3.0
Both show `gated: "auto"`, meaning access is granted immediately on
accepting, no manual repo-owner review wait. No code or pin can substitute
for this step.

**Takeaway.** A metadata-only HF API call (`model_info`,
`list_repo_files`, etc.) is not a valid proxy for "can this token actually
download this gated file" -- gated-repo enforcement in `huggingface_hub`
0.25.2 only triggers on the real download path. Any future
connectivity/smoke check for gated content should exercise the same call
the real code makes, not a cheaper-looking substitute that happens to
return 200/OK for an unrelated reason.
