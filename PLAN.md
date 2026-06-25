# Handoff notes (read this first)

This file exists so a fresh Claude Code session (e.g. one with network
access this sandbox didn't have) can pick up exactly where the previous
session left off, without relying on local-only state like
`/root/.claude/plans/*.md` or an in-session task list -- neither of those
survive a session/environment switch, only what's committed to this repo
does.

## Task checklist (status as of this commit)

- [x] #1 Repo skeleton (dirs, .gitignore, requirements.txt, .env.example)
- [x] #2 `pipeline/config.py` + `config/pipeline.yaml`
- [x] #3 `pipeline/db.py` (SQLite schema + state machine)
- [x] #4 `pipeline/podcastindex_client.py`
- [x] #5 `pipeline/audio.py` (ffmpeg wrappers)
- [x] #6 `pipeline/vad.py` (Silero VAD wrapper)
- [x] #7 `pipeline/diarize.py` (pyannote wrapper)
- [x] #8 `pipeline/cluster.py` (cross-episode speaker clustering)
- [x] #9 `pipeline/segment.py` (clip segmentation algorithm)
- [x] #10 `pipeline/quality.py` (discard taxonomy)
- [x] #11 `pipeline/asr.py` (faster-whisper wrapper)
- [x] #12 `pipeline/manifest.py` (JSONL construction + schema validation)
- [x] #13 `pipeline/storage.py` (R2 boto3 client)
- [x] #14 `pipeline/costs.py` (cost ledger + budget guardrails)
- [x] #15 `pipeline/heartbeat.py` + `pipeline/logging_utils.py`
- [x] #16 `pipeline/ingest.py` + `pipeline/pipeline_runner.py` + `run_pipeline.py`
- [ ] #17 **in progress** `scripts/`:
  - [x] `select_podcasts.py`
  - [x] `smoke_test.py`
  - [x] `partition_episodes.py`
  - [x] `package_code.py`
  - [ ] `bootstrap_pod.py` -- not started
  - [ ] `poll_status.py` -- not started
  - [ ] `merge_shards.py` -- not started
  - [ ] `report.py` -- not started
  - [ ] `validate_manifest.py` -- not started
- [ ] #18 `tests/` (test_db, test_segment, test_quality, test_manifest, test_costs, test_cluster) -- not started
- [ ] #19 `infra/bootstrap.sh` + `Dockerfile` -- not started
- [ ] #20 Run full local test suite, fix failures -- not started
- [ ] #21 Confirm network egress is unblocked, then run real PodcastIndex/RunPod/HF/R2 checks -- **this is almost certainly the first thing to do in the new session**

## Verification done so far (no GPU, no network, in the build sandbox)

- Every `pipeline/` module's function signatures were cross-checked against
  actual call sites before writing dependent code (no guessed APIs).
- `pipeline_runner.py` + `run_pipeline.py` were exercised with a full
  integration smoke test: real ffmpeg transcode/probe/clip-extraction, real
  SQLite state-machine transitions, real segment/cluster/quality interval
  math, real local-fallback export -- with only the four GPU/network leaf
  calls monkeypatched (`ingest.download_episode_audio`, `vad.run_vad`,
  `vad.frame_speech_probabilities`, `diarize.diarize`,
  `asr.transcribe_clips_for_episode`). All four scenarios passed: fresh full
  run, no-op resume on an already-`done` episode, induced mid-pipeline
  failure with prior-stage work preserved, and correct resume-from-failure
  on retry (not from `queued`).
- `scripts/select_podcasts.py`'s selection/bin-packing math, db registration,
  and `scripts/partition_episodes.py`'s LPT-greedy shard assignment were
  each verified against synthetic/local data (see git history for the exact
  test snippets run -- they weren't committed since they were throwaway
  verification, not test suite deliverables; #18 will add the real
  pytest-based equivalents).
- `scripts/package_code.py` was run end-to-end (`--no-upload`) and produces
  a clean tarball of `pipeline/` + `config/` + `requirements.txt` +
  `run_pipeline.py`, excluding `__pycache__`.

## Known gap: `pipeline/runpod_client.py` is UNVERIFIED against live docs

This sandbox's egress proxy hard-blocked `rest.runpod.io` *and*
`docs.runpod.io` (confirmed via direct 403 policy-denial responses, not a
transient error), so the exact endpoint paths and request/response field
names in `pipeline/runpod_client.py` reflect RunPod's published REST API v1
conventions (Bearer auth, `/pods` and `/gpuTypes` resources) from training
knowledge, not a verified live call. **Before writing `bootstrap_pod.py`'s
pod-creation call (which spends real money) or running
`scripts/smoke_test.py`'s full RunPod check for real, confirm the actual
endpoint shapes against RunPod's live API reference** (e.g. fetch
`https://rest.runpod.io/v1/openapi.json` or the docs site) and fix
`pipeline/runpod_client.py` if anything doesn't match. The field names are
deliberately isolated in one place (`RunPodClient.create_pod`'s body dict
and the per-method paths) specifically to make this a small, contained fix.

## Immediate next steps in a session with network access

1. Run `python3 scripts/smoke_test.py --check-network-only` first. If any of
   `api.podcastindex.org`, `rest.runpod.io`, `huggingface.co`, or the R2
   endpoint are still unreachable, stop and report that rather than
   retrying or working around it -- per the original plan's Step 0.
2. Populate `.env` (copy from `.env.example`) with real credentials:
   PodcastIndex key/secret, RunPod API key, R2 account id/keys/bucket, HF
   token (with the pyannote 3.1 + embedding model gated-model agreements
   accepted on huggingface.co).
3. Run `python3 scripts/smoke_test.py` (full mode) and fix anything that
   fails -- this is also where `runpod_client.py`'s endpoint shapes get
   their first real-world check; fix them here if they're wrong before
   anything spends money.
4. Resume Task #17: write `bootstrap_pod.py`, `poll_status.py`,
   `merge_shards.py`, `report.py`, `validate_manifest.py`.
5. Continue with #18 (tests/), #19 (infra/), #20 (run suite), matching the
   Staged rollout section below.
6. **Before creating the first real RunPod pod (the smoke-test pod), present
   GPU type/cloud-tier/estimated rate/expected cost to the user and get
   explicit go-ahead** -- this is the user's explicit instruction from the
   original task ("notify before the first dollar is spent"), and it still
   applies in the new session.

---

# Original plan (as approved before this build started)

The plan below is the design document this build has been following since
session start. It hasn't been edited to reflect implementation decisions
made along the way (e.g. `runpod_client.py`'s existence, since the original
repo layout didn't list a dedicated RunPod client module -- it's referenced
inline within `bootstrap_pod.py`/`poll_status.py`/`smoke_test.py` instead;
that's a reasonable implementation refinement, not a deviation worth
re-litigating). Treat the "Handoff notes" section above as the
authoritative status; treat everything below as the original design intent.

## Context

The repo contains only the take-home spec (`FILE_3000.pdf`) — this is a greenfield build. The spec: convert public podcast audio (via PodcastIndex) into a LibriLight-style dataset — short, clean, single-speaker clips with a JSONL manifest — within a 24h elapsed window and a $100 reimbursable budget. Evaluation weighs throughput/cost discipline, data quality (especially clean single-speaker segmentation), scalability, operational maturity (checkpointing, metrics, failure handling), cost reasoning, reproducibility, and pragmatism.

Locked-in decisions (not being relitigated): RunPod RTX 3090 for compute, Cloudflare R2 for storage, pyannote for diarization + cross-episode speaker clustering (speaker-ID accuracy is the top technical priority — this is training data for an audio-separation model — balanced against throughput), faster-whisper `base` for ASR (used as a quality-filtering signal and optional transcript, not the core deliverable), Silero VAD for silence/non-speech removal, ffmpeg for preprocessing, SQLite for resumable checkpointing, full JSONL manifest per the spec schema, explicit cost tracking against the $100 cap. User wants build **and** execute, with explicit notification before the first dollar is spent, and a written explanation of how every major choice trades off throughput/performance against cost.

**Research finding that reshapes the cost story**: RunPod RTX 3090 runs ~$0.19–0.30/hr (Community Cloud, fluctuates) to ~$0.44/hr (Secure Cloud). At those rates, $100 buys 200+ GPU-hours — far more than the 24h window allows. **Money is not the binding constraint here; wall-clock time is** (24h hard ceiling, ~6–8h active work expected). This shapes every throughput/cost tradeoff below: optimize for finishing reliably within the time window via a staged, checkpointed rollout, not for squeezing GPU-dollars.

This revision adds two things on top of the previously-approved design: (1) a grounding in how LibriSpeech and Libri-Light — the two canonical precedents for "LibriLight-style" datasets — actually solved segmentation, quality filtering, and speaker ID, so design choices here are validated against (or explicitly contrasted with) real prior art rather than invented from scratch; and (2) explicit accounting for the cost of using Claude Code itself to build and run this pipeline, as a transparency line item distinct from the $100 infra cap the spec actually asks for.

## Prior art: LibriSpeech and Libri-Light

Both corpora are derived from LibriVox audiobooks and are the direct namesake/precedent for "LibriLight-style" data. Researched via web search and the Libri-Light GitHub data-prep README (the original papers on arXiv/openslr.org/danielpovey.com all returned HTTP 403 to WebFetch from this sandbox, so figures below come from search-result summaries and the GitHub README, not a direct read of the primary PDFs — caveated inline where a number isn't independently confirmed).

**LibriSpeech (Panayotov et al., ICASSP 2015, ~1000h):**
- Segmentation came from two-pass Kaldi forced alignment (a triphone GMM-HMM model discriminatively trained with Boosted MMI, MFCC+LDA+global-STC features) against a per-chapter biased language model — i.e. alignment-driven, not VAD-driven, because they had the book text as ground truth.
- Quality filtering used WADA-SNR-style estimation; one commonly cited figure is a ~20dB SNR threshold filtering out roughly 25% of sentences for the "clean" split. Treat this as approximate/secondhand — search results explicitly flagged it as referenced mostly in derivative-corpus writeups rather than confirmed directly from the 2015 paper's methodology section.
- Speaker ID was never inferred — each audiobook has one documented LibriVox narrator, so speaker identity is just metadata lookup.

**Libri-Light (Kahn et al., ICASSP 2020, ~60,000h):**
- VAD via a wav2letter++ CTC-trained TDS acoustic model producing frame-level SPEECH/NONSPEECH posteriors (conceptually the same role Silero VAD plays in this pipeline, different model).
- Per-file SNR computed from the ratio of VAD-derived speech-frame power to non-speech-frame power, using what their own docs call "a dataset specific threshold" — and they explicitly recommend tuning it by inspecting per-file SNR histograms rather than hardcoding a number. **This is a direct precedent for this plan's existing approach of tuning `match_threshold` and quality-filter floors during the Stage 1 human-checkpoint rather than fixing them upfront.**
- Segmentation concatenates consecutive VAD-positive chunks into ~60-second sequences — a memory/training-pipeline-driven choice, not utterance-level. This is a real contrast, not a precedent: the take-home spec requires clip-level granularity (<10s typical, <30s tail), which is a strictly harder segmentation problem than Libri-Light solved.
- Speaker ID again comes entirely from LibriVox per-book/per-reader metadata (~7,000 known speakers) — Libri-Light was not designed to do diarization or cross-recording speaker re-identification at all.
- The corpus is organized into three explicit scale/quality tiers — small (577h/35GB), medium (5,193h/321GB), large (51,934h/3.05TB) as unlab-600/unlab-6k/unlab-60k. This is a useful precedent for the at-scale section below: a tiered "clean-strict vs. high-recall" output strategy, rather than one undifferentiated pool, is exactly how the corpus most directly analogous to this task's goal already chose to scale.

**Why this matters for this plan specifically:** neither corpus had to solve the part of this task that's actually hard. Both get speaker identity for free from known audiobook-narrator metadata, and neither has multiple speakers talking in the same recording. Podcasts have multiple speakers per episode and hosts/guests that recur or change across episodes, with zero metadata shortcut — so diarization + cross-episode speaker clustering (`pipeline/diarize.py` + `pipeline/cluster.py`) is the one component of this pipeline with no direct precedent in either reference corpus. This isn't a reason to change the design (pyannote diarization + cosine-similarity clustering is still the right call), but it is the single highest-risk, least-validated piece of the system, and `LIMITATIONS.md`/the speaker-ID section of `WRITEUP.md` should say so plainly rather than implying this is a solved problem just because it superficially resembles LibriLight. Concretely, this research:
- Validates the VAD-first, confidence-floor-driven segmentation approach already planned (`pipeline/vad.py`, `pipeline/segment.py`) as consistent with how Libri-Light actually built its pipeline.
- Validates tuning thresholds empirically during the Stage 1 checkpoint (already planned) rather than hardcoding them — Libri-Light's own docs recommend exactly this.
- Suggests adding a lightweight SNR-style energy-ratio check (already partially covered by the `silence_or_low_energy` RMS-floor filter in `pipeline/quality.py`) as a named, explicit echo of the WADA-SNR/Libri-Light SNR precedent — same filter, but the writeup should name the precedent instead of presenting it as a from-scratch heuristic.
- Motivates explicitly modeling the at-scale (10,000+ hour) discussion in `COST_REPORT.md`/`WRITEUP.md` on Libri-Light's tiered-subset structure: e.g. a "strict" tier (high speaker-ID confidence, conservative clustering threshold) vs. a "broad" tier (looser thresholds, higher yield, more manual-review risk) instead of one undifferentiated output.
- Sharpens the speaker-ID failure-mode writeup (already planned in `pipeline/cluster.py`'s documented failure modes) by naming explicitly that this is the part of the task with no equivalent in the reference corpora — strengthening the "documented failure modes" requirement from spec Requirement 4 rather than changing any code.

No code/architecture changes result from this section — it's a validation-and-framing layer on the existing design, surfaced in `WRITEUP.md` (new "Prior art" subsection) and `LIMITATIONS.md` (sharper framing of the speaker-ID caveat).

## Claude/agent compute cost accounting

The spec's $100 cap and cost-analysis requirements are about cloud infra (RunPod/R2/PodcastIndex) — they say nothing about the cost of using Claude Code to build and operate the pipeline itself. That's a real dollar cost the user is incurring and asked to have accounted for, but it is conceptually separate from the reimbursable infra budget and must not be merged into the $100 figure or the `cost_events` ledger (which is specifically scoped to infra in Requirement/Cost-Analysis terms).

Design:
- Add a short, clearly-labeled **"Agent compute cost (informational, not part of the $100 infra budget)"** section to `WRITEUP.md` (not `COST_REPORT.md`, to keep the spec's actual deliverable format clean) explaining: this Claude Code session was used to design, build, debug, and operate the pipeline; its token cost is billed separately from and outside the RunPod/R2/PodcastIndex budget; it's disclosed here for transparency only.
- Pricing basis: this session runs on Claude Sonnet 4.6 — $3.00 / MTok input, $15.00 / MTok output (current published rates). Prompt-caching is in effect for long-running sessions like this one (cache reads ≈0.1× base input price; cache writes ≈1.25–2× base input price depending on TTL), which materially lowers effective cost on a long single-session build like this one versus naive input-token pricing.
- **Methodology caveat, stated explicitly in the writeup**: this agent has no mid-session way to query its own cumulative token usage (no `usage` introspection tool is exposed inside the harness), so an exact metered dollar figure can't be self-computed. The writeup will present whichever of these the user can supply, in order of preference:
  1. The actual figure from the Claude Code CLI's own `/cost` command, run locally by the user — the authoritative source if available.
  2. Failing that, an explicit order-of-magnitude estimate (e.g. "session involved on the order of N tool calls / web fetches / file reads across ~X hours of active work; at Sonnet 4.6 rates this is roughly in the $Y–$Z range") clearly labeled as an estimate, not a metered number.
- No new code module for this — it's a documentation-only addition (`WRITEUP.md` section), since there's nothing for `pipeline/costs.py` to ledger (that module's `cost_events` table stays scoped to GPU/storage/network as already designed, per the spec's actual schema).

## Scale revision: targeting 100+ clean hours via horizontal pod parallelism

Mid-build, the target was raised from ~6-10 raw hours to **100+ hours of clean, usable, speaker-attributed audio**, with speaker-ID accuracy held to the same bar (no loosening of clustering thresholds or quality filters to hit the number). This changes the throughput math enough to need a real architectural addition — horizontal scaling across multiple RunPod pods — layered on top of the single-pod design below, not a replacement for it.

**Why a single pod is no longer a safe bet.** Researched pyannote's own published figure (speaker-diarization-3.0/3.1 model card): RTF ≈ 2.5% (≈40x real-time) for segmentation+embedding on a V100 GPU. But a GitHub issue against the same repo reports real-world throughput up to ~50x slower than that figure in practice — so the credible planning range for the GPU-bound diarization stage spans roughly 1x to 40x real-time, a 40x spread. faster-whisper has no confirmed official RTF for the `base` model specifically (only large-v2/v3 benchmarks exist); Silero VAD is confirmed comfortably non-bottleneck (30-100x+ real-time even on CPU alone). Given that spread, committing to a single pod and hoping it lands fast enough within 24h is a real risk to the actual optimization metric — exactly the failure mode the original plan was already designed to avoid ("the real risk... is a late-stage failure forcing a restart").

**Yield assumption.** Clean usable speech as a fraction of raw podcast audio (after VAD, overlap exclusion, ads/intro/outro/music discards) is planned at ~65% (range 55-75%, talk/interview shows), pending Stage 1's real discard-reason counts. To net 100+ clean hours with margin, the candidate raw-hour pool targets **~150-200 raw hours** (100 / 0.65 ≈ 154, rounded up for hedge) — roughly 15-25x the original target, not just the headline 10x the clean-hour number might suggest.

**Design: measure first, then size parallelism — same philosophy as threshold-tuning, applied to pod count.**
1. Stage 1 (unchanged below) still runs on exactly one pod and still exists primarily to validate quality/speaker-ID — but it now also doubles as the empirical throughput calibration the pod-count decision depends on: measure real wall-clock-per-raw-hour for the slowest stage (diarization) on this actual code, this actual hardware, real podcast audio — not a borrowed benchmark number.
2. From that measurement, compute `pods_needed = ceil(remaining_raw_hours / (measured_hours_per_wallclock_hour × stage2_wallclock_budget))`, where `stage2_wallclock_budget` is deliberately capped below the full 24h (e.g. ~10-14h) to leave margin for the smoke test, Stage 1 inspection, threshold tuning, merge, and report-writing. Round up with a +20-30% safety margin — over-provisioning a pod or two is cheap insurance, not waste, given RunPod's per-hour rate.
3. Launch that many pods in Stage 2, each assigned a disjoint shard of queued episodes (see `scripts/partition_episodes.py` below) — no cross-pod coordination needed during processing, only at the end.
4. If the multi-pod watchdog (extended `scripts/poll_status.py`) shows aggregate progress falling behind pace partway through, the recommendation to launch additional pods against not-yet-claimed episodes is surfaced to the user rather than auto-executed — launching more paid compute mid-run is a real spend decision, consistent with the existing notify-before-spend posture, even though the absolute dollar amounts involved stay small.
5. Stop once the aggregate running clean-hour count (summed across pod heartbeats) crosses 100+ with margin; finish in-flight episodes, then merge.

**Speaker-ID accuracy under sharding — the one real correctness hazard, and how it's closed.** If a single podcast's episodes get split across two pods, each pod's *local* incremental speaker-ID numbering (e.g. `podcast123_speaker_002`) is only locally consistent — two pods can independently mint the same ID string for two different actual people. The fix: per-episode diarization output and embeddings (`local_speaker_segments`) are durable and shard-safe (keyed by globally-unique `episode_id` + local label, no collision risk), but every pod's own `speakers`/`clips.speaker_id` assignments are treated as **provisional** the moment more than one pod is in play. After all shards finish, a new centralized merge step (`scripts/merge_shards.py`) downloads each pod's exported SQLite DB from R2, merges `local_speaker_segments`/`clips`/`cost_events` (safe — globally unique keys throughout), then re-runs `pipeline/cluster.py`'s existing re-clustering logic **once, from scratch, per podcast, over the merged embeddings** — discarding every shard-local `speakers`/`clips.speaker_id` value and replacing it with one centrally-computed, globally-consistent answer. This is the *same* agglomerative-clustering function the single-pod design already used for periodic re-clustering (it already operated on "all persisted embeddings for a podcast," regardless of provenance) — applied uniformly to every podcast post-merge, not just split ones, for simplicity and uniform correctness. Net effect: parallelism changes *where* diarization runs, never *how* final speaker identity is decided — accuracy is preserved by construction, not by hoping the shards happen to agree.

**Partitioning unit.** Episodes (not whole podcasts) are bin-packed across shards by reported duration for load balance — `episodes.assigned_shard` (new column, set once by `scripts/partition_episodes.py` before any pod starts) tells each pod which episodes it owns (`WHERE assigned_shard = ? AND stage = 'queued'`). Allowing a podcast to span shards (rather than forcing one-podcast-per-pod) gives much better load balancing given unknown-until-queried episode durations, and costs nothing extra in correctness since the merge step always recomputes clustering centrally regardless.

**Revised cost/storage numbers.** Even a pessimistic 12 pods × 12h × $0.30-0.44/hr ≈ $43-63 — still under the $100 cap, but no longer a trivial fraction of it the way the original 6-10 raw-hour target was. Money is still the looser constraint than wall-clock time, but at this scale it's worth tracking deliberately rather than treating as a rounding error — pod count should be sized to the measured throughput target, not padded arbitrarily. R2 storage for 100+ clean hours of flac-encoded speech-only audio (~60-80kbps effective after VAD trimming) is still only on the order of a few GB — likely still inside R2's 10GB free tier, but no longer assumed $0 by inspection alone; report the real measured bytes either way. Per-pod local disk stays bounded regardless of total corpus size because raw audio is processed-and-deleted per episode (never bulk-downloaded upfront), and each pod only ever holds its own shard's working set, not the whole candidate pool.

**Podcast selection at this scale.** `scripts/select_podcasts.py` widens from "4-6 shows" to "as many interview/talk-style shows as needed to assemble a ~150-200 raw-hour candidate pool" — expected to land somewhere around 10-20 shows depending on real average episode length and catalog availability, which can't be pinned down further until the (currently network-blocked) PodcastIndex query actually runs. Longer-episode shows are preferred when there's a choice, since intro/outro/ad overhead is roughly fixed per episode and amortizes better over longer episodes — directly improving yield.

No other locked-in decision changes: same diarization model, same VAD, same quality filters, same discard taxonomy, same clustering math (just invoked centrally-and-from-scratch once at merge time, in addition to its existing per-pod incremental/periodic use), same notify-before-first-dollar-spent gate at the smoke test.

## Blocking prerequisite: network egress

This sandbox's egress proxy currently allowlists only PyPI/npm/GitHub and explicitly denies `api.podcastindex.org`, `rest.runpod.io` (and RunPod's other API/proxy hosts), `huggingface.co`, and Cloudflare (confirmed via direct 403 policy-denial responses from the proxy, not a transient error). The user has chosen to reconfigure this environment's network policy (via Claude Code on the web environment settings) rather than build-only-and-handoff.

**Step 0 of execution, before anything else network-dependent**: re-run a connectivity check (`scripts/smoke_test.py --check-network-only`) against `api.podcastindex.org`, `rest.runpod.io`, `huggingface.co`, and the R2 S3 endpoint. If still blocked (e.g. because the policy change requires a fresh session/environment rather than taking effect live), stop and surface that to the user rather than retrying or working around it — per the proxy's own guidance, policy denials get reported, not routed around.

Everything in "Phase A" below (all code, all local tests) is unaffected by this and can proceed immediately regardless of network state, since it only needs PyPI/GitHub.

## Repo layout

```
pipeline/
  config.py            # load pipeline.yaml into a validated config object
  db.py                # SQLite schema + state-machine helpers (resumability)
  podcastindex_client.py  # PodcastIndex auth (SHA1(key+secret+timestamp)) + search/episodes calls
  ingest.py             # download episode audio, ffprobe duration, register in DB
  audio.py              # ffmpeg wrappers: transcode to 16k mono wav, encode clips to flac
  vad.py                # Silero VAD -> speech sub-segments with confidence
  diarize.py            # pyannote/speaker-diarization-3.1 -> local speaker turns + embeddings
  cluster.py            # cross-episode speaker clustering (centroids, matching, periodic re-cluster)
  segment.py            # clip-segmentation algorithm (the core duration-distribution logic)
  quality.py            # discard-reason taxonomy / automated vs heuristic filters
  asr.py                # faster-whisper base -> transcript + no_speech_prob + avg_logprob
  manifest.py           # JSONL row construction + schema validation
  storage.py            # R2 client (boto3, S3-compatible): put/get/list for clips, manifest, status
  costs.py              # cost_event ledger, running total, budget-cap checks (infra-only scope)
  heartbeat.py          # writes status.json to R2 + serves it over an HTTP status port
  pipeline_runner.py    # per-episode stage-machine driver
  logging_utils.py      # structured logging to file, periodic R2 sync, secret redaction
run_pipeline.py          # CLI entrypoint
config/pipeline.yaml      # all tunables (thresholds, model names, budget/time caps)
config/podcasts.json      # selected podcasts (output of select_podcasts.py, checked in for reproducibility)
scripts/
  select_podcasts.py      # PodcastIndex search/filter -> config/podcasts.json (sized to raw-hour pool target)
  smoke_test.py            # connectivity + auth checks for all 4 external services, no GPU
  partition_episodes.py    # bin-packs queued episodes across N shards by reported duration
  package_code.py          # tars pipeline/+config/+requirements.txt, uploads to R2
  bootstrap_pod.py         # calls RunPod API to create N pods, each with a --shard-id + env vars
  poll_status.py           # polls R2 status.json for all active pods; aggregates; can call stop/terminate
  merge_shards.py          # downloads each pod's pipeline.db from R2, merges, re-runs cluster.py from scratch per podcast
  report.py               # SQLite -> PROCESSING_SUMMARY.md, COST_REPORT.md, sample manifest
  validate_manifest.py    # JSONL schema + duration-distribution sanity checks
infra/
  bootstrap.sh             # the actual RunPod start command (apt/pip installs, fetch code, run)
Dockerfile                 # reproducibility reference (not what's deployed to RunPod directly)
tests/                     # test_db.py, test_segment.py, test_quality.py, test_manifest.py,
                            # test_costs.py, test_cluster.py — all pure-Python, no GPU, no network
.env.example, .gitignore, requirements.txt
README.md, WRITEUP.md, PROCESSING_SUMMARY.md, COST_REPORT.md, LIMITATIONS.md
```

## Database schema (resumability backbone)

SQLite at `work/pipeline.db`, WAL mode, single-writer. Tables:

- **`podcasts`**: `podcast_id, feed_id, title, feed_url, language, episode_count_total, selected_at, selection_reason`.
- **`episodes`**: `episode_id, podcast_id, pi_episode_id, title, source_url, published_at, duration_seconds_reported, duration_seconds_actual, assigned_shard, local_raw_path, local_wav_path, stage, failed_stage, attempt_count, last_error, raw_seconds, usable_seconds, created_at, updated_at`. `assigned_shard` is set once by `scripts/partition_episodes.py` before any pod starts (null in the single-pod/Stage-1 case); each pod's queue query filters `WHERE assigned_shard = ? AND stage = 'queued'`. `stage` is an ordered state machine: `queued → downloading → downloaded → transcoding → transcoded → vad_running → vad_done → diarizing → diarized → clustering_done → segmenting → segmented → asr_running → asr_done → quality_filtering → exporting → uploading → done`, with a `failed` state carrying `failed_stage` so retries resume from the failure point, not from `queued`.
- **`speakers`**: global per-podcast identities — `speaker_id, podcast_id, local_label_seq, centroid_embedding (BLOB), centroid_dim, embedding_count, total_speech_seconds, first_seen_episode, last_seen_episode, created_at, updated_at`. Treated as provisional per-shard during multi-pod Stage 2 — authoritative only after `merge_shards.py`'s centralized recompute (see Cross-episode speaker clustering below).
- **`local_speaker_segments`**: raw per-episode diarization output, persisted independently so re-clustering never re-runs diarization — `episode_id, local_label, start_seconds, end_seconds, embedding, resolved_speaker_id`. This table is the durable, shard-safe source of truth (globally-unique `episode_id` keys mean merging across shards is collision-free); `speakers`/`clips.speaker_id` are always-recomputable derived views from it.
- **`clips`**: one row per candidate clip including discarded ones (for the discard-reason report) — `clip_id, episode_id, podcast_id, start_seconds, end_seconds, duration_seconds, speaker_id, utterance, vad_confidence, overlap_detected, music_detected, no_speech_prob, avg_logprob, discard_reason, audio_path, local_flac_path, uploaded, created_at`.
- **`cost_events`**: append-only ledger — `ts, category (gpu_compute|r2_storage|r2_class_a_ops|r2_class_b_ops|egress|other), description, amount_usd, related_episode_id, metadata_json`. Scoped to infra costs only (Claude/agent compute cost is documentation-only, see above — not ledgered here). Per-pod during Stage 2; `merge_shards.py` concatenates all shards' ledgers for the final `COST_REPORT.md`.
- **`run_meta`**: key/value (pod_id, shard_id, pod_started_at, budget caps, etc.).

Resumability rule: every `pipeline_runner.run_<stage>(episode_id)` function is a no-op if the episode's `stage` is already past that stage; on exception it records `failed_stage`/`last_error` and the driver moves on to the next episode rather than crashing the run. Clip upload resumes via `WHERE uploaded = 0` — no duplicate R2 writes or manifest rows on rerun.

## Clip segmentation algorithm (`pipeline/segment.py`)

1. Build an overlap-exclusion mask from any time range where ≥2 distinct diarization speakers overlap — excluded from clip candidates entirely (crosstalk requirement).
2. Intersect each single-speaker diarization turn with Silero VAD speech sub-segments, subtracting the overlap mask → "clean single-speaker speech intervals."
3. Per interval: if duration ≤ 10s, emit as one clip (subject to a `min_clip_duration` ≈1–1.5s floor, else discard `too_short`). If 10–30s, look for a natural pause point (via Silero's frame-level speech probabilities, not just merged segments) in the middle portion and split there when it improves the duration-distribution mix; if no clean pause exists, keep the whole interval as a legitimate long-tail clip. If >30s, repeatedly cut at the best pause point nearest the 30s cap; if a stretch has no detectable pause at all, force a hard cut at exactly 30s (rare, logged as a soft quality note, not a discard).
4. Bias cut-point choice using a running per-episode histogram against target bucket ratios (e.g. ~70% <10s / ~25% 10–20s / ~5% 20–30s, configurable) so the aggregate distribution matches the spec's "mostly under 10s, meaningful long tail to 30s" — a local greedy heuristic, documented as such, not a global optimizer.
5. Emit `CandidateClip` rows into `clips` (discard bookkeeping for content reasons happens later in `quality.py`, kept centralized).

## Cross-episode speaker clustering (`pipeline/cluster.py`)

- **Per episode** (`diarize.py`): run pyannote diarization once with `return_embeddings=True` — confirmed directly from the installed pyannote.audio 3.3.2 source (`pipelines/speaker_diarization.py`, `SpeakerDiarization.apply()`), not assumed from docs: this returns `(diarization, embeddings)` where `embeddings[i]` is *already* the representative centroid for local speaker `diarization.labels()[i]`, aggregated by pyannote's own clustering step over that speaker's chunks for the whole episode (with `embedding_exclude_overlap=True` configured so crosstalk frames don't pollute it). **This supersedes the original plan of a separate `pyannote/embedding` pass + top-K-longest-turn selection** — one fewer GPU model loaded and one fewer pass per episode, since pyannote's diarization step was already computing per-chunk embeddings internally to do its own clustering, and `return_embeddings=True` just exposes that result instead of discarding it. A local speaker's centroid is only kept if their total speech time across all turns in the episode clears `clustering.min_local_speaker_seconds_for_embedding` (default 1.5s — same threshold as originally planned, now applied per-local-speaker-per-episode rather than per-turn) and the row isn't one of pyannote's documented zero-padded placeholder rows (emitted when frame-level speaker-counting briefly overcounts beyond what clustering produced centroids for). Turns are still persisted to `local_speaker_segments` regardless of whether their speaker cleared the embedding floor (diarization is never re-run).
- **Incremental matching** (same pass): compare each local speaker's pyannote-provided centroid via cosine similarity against existing global `speakers` centroids *for that podcast only* (matching is podcast-scoped, matching the `podcast123_speaker_002` ID format); above `match_threshold` (default 0.75, tuned during the Stage 1 validation checkpoint) → assign existing ID and update its centroid as a duration-weighted running average; below threshold → mint a new global speaker ID. This step is O(existing speakers for that podcast) per episode — cheap and linear.
- **Periodic re-clustering** (every N=5 episodes per podcast, or on demand): re-run agglomerative clustering (cosine distance, average linkage) over all persisted embeddings/local centroids for that podcast — catches drift the incremental pass misses (e.g. same speaker, different recording conditions across episodes). Operates on embedding vectors (KB-scale), not audio, so it stays fast even at hundreds of episodes; merges get reconciled back into `speakers` (and any already-written `clips.speaker_id` get corrected, logged as a traceable event).
- **From-scratch recompute mode** (same underlying agglomerative-clustering function, new entry point for multi-pod Stage 2): `recluster_podcast_from_scratch(podcast_id, db)` takes the full set of merged `local_speaker_segments` for a podcast — regardless of which pod produced them — and (re)builds `speakers`/`clips.speaker_id` from nothing, ignoring any shard-local provisional IDs. `scripts/merge_shards.py` calls this once per podcast after merging all pods' DBs, which is what makes parallel processing safe for speaker-ID accuracy: per-shard `speaker_id` strings can collide in meaning across pods (two pods may both mint `podcast123_speaker_002` for different actual people), but `local_speaker_segments` never collides (globally-unique `episode_id` keys), so the merge always has a clean substrate to recompute from.
- **Known failure modes to document explicitly**: timbre-similar speakers may merge (mitigated by conservative threshold tuning — prefer false splits over false merges, since splits are less damaging to an audio-separation training set); the same speaker can split across recording-condition changes (periodic re-clustering partially mitigates, not guaranteed); short turns produce no usable embedding (speaker left unassigned, not fatal — manifest allows null `speaker_id`); one-off guests correctly get a never-reused ID; clustering inherits any upstream diarization errors. **This is also the one component of the pipeline with no precedent in LibriSpeech or Libri-Light (see Prior art section) — both reference corpora get speaker identity for free from audiobook-narrator metadata and never had to diarize or re-identify speakers across recordings, so this module carries more genuine uncertainty than the rest of the pipeline and should be flagged as such rather than presented as a solved problem.**

## Quality filters / discard taxonomy (`pipeline/quality.py`)

| discard_reason | method | automated vs heuristic |
|---|---|---|
| `too_short` | duration floor | heuristic threshold |
| `silence_or_low_energy` | RMS energy floor (catches VAD false positives); same role as the WADA-SNR / VAD-derived SNR filters in LibriSpeech/Libri-Light (see Prior art) | heuristic |
| `overlap_detected` | pyannote overlap regions + edge-trim tolerance | automated + heuristic tolerance |
| `music_detected` | spectral flatness/harmonic ratio **corroborated by** whisper `no_speech_prob` | heuristic (no dedicated music classifier in the stack — flagged as the weakest filter) |
| `low_asr_confidence` | whisper `avg_logprob`/`no_speech_prob` as a corroborating signal, not sole trigger | automated signal, heuristic application |
| `intro_outro_position` | first/last ~30s of episode + spectral corroboration | heuristic |
| `ad_segment_heuristic` | weak position + spectral-change heuristic | heuristic only — explicitly the least reliable filter, called out plainly in `LIMITATIONS.md` rather than overclaimed |
| `repeated_boilerplate` | cross-episode near-duplicate transcript/audio fingerprint (stretch goal, time-permitting only) | automated-ish, optional |
| `vad_low_confidence` | Silero VAD confidence floor | automated |

This table goes near-verbatim into `WRITEUP.md`'s filters section to satisfy the spec's explicit "explain which filters are automated vs heuristic" requirement, and the music/ad caveat goes into `LIMITATIONS.md`.

## RunPod execution & monitoring (no SSH, no pod-log API)

RunPod's API supports pod lifecycle (create/list/stop/terminate) over HTTPS but **does not** expose container logs via API — confirmed via RunPod's own issue tracker. Design avoids depending on SSH or log retrieval entirely:

1. **Secrets**: `.env` (gitignored) holds `PODCASTINDEX_API_KEY/SECRET`, `RUNPOD_API_KEY`, `R2_ACCOUNT_ID/ACCESS_KEY_ID/SECRET_ACCESS_KEY/BUCKET_NAME`, `HF_TOKEN`, `BUDGET_CAP_USD`, `TIME_CAP_HOURS`. `.env.example` documents names only. Secrets reach the pod exclusively via RunPod's own env-var injection at pod creation — never via the code tarball or git. `logging_utils.py` redacts anything matching known secret patterns before any log content is synced to R2, as defense in depth.
2. **Bootstrap**: no custom Docker image pushed to a registry (avoids needing Docker Hub credentials and an extra blind-build-failure surface). Use RunPod's official prebuilt PyTorch+CUDA template image; `infra/bootstrap.sh` is the pod start command: `apt-get install ffmpeg`, fetch the code tarball from R2 (pushed beforehand by `scripts/package_code.py`), `pip install -r requirements.txt`, run `run_pipeline.py`. The root `Dockerfile` documents the equivalent for reproducibility elsewhere (per spec requirement 7) but isn't what's deployed to RunPod for this trial.
3. **Self-reporting, dual channel**: the pipeline writes a status/heartbeat JSON to R2 after every episode (or on a timer) — stage, episodes done, clips produced, elapsed GPU time, running cost, last error. As cheap redundant insurance, also serve the same status dict over a lightweight `http.server` thread on RunPod's exposed proxy port, so a status check doesn't depend solely on R2 write success.
4. **Smoke test before any real batch**: launch the pod once with `--smoke-test-mode` (one short episode end-to-end), confirm a heartbeat appears in R2 within a generous timeout (~10 min, covers first-time model downloads) and that a real clip + manifest row land in R2, then **stop** (not terminate) the pod pending inspection. If no heartbeat appears in that window, that's the signal something failed silently before Python/logging started (apt/pip phase) — fall back to RunPod's web dashboard logs manually rather than guessing.
5. **Watchdog, extended for N pods**: `scripts/poll_status.py` polls every active pod's R2 status object (keyed by `pod_id`/`shard_id`) on an interval, aggregates clean-hours/cost/elapsed across all of them, and has authority to call `stop_pod`/terminate on any individual pod if its heartbeat goes stale for ~15-20 minutes or either cap (cost/time) is clearly exceeded — never solely dependent on a pod successfully self-terminating. If aggregate pace is falling behind the 100+ clean-hour target partway through, it surfaces a "launch K more pods" recommendation rather than auto-launching them (see Scale revision section) — spend decisions beyond the original smoke test stay visible to the user.
6. **Notify-before-spend checkpoint**: the first action that incurs real cost is the smoke-test pod creation. Before that call, present GPU type/cloud-tier/estimated rate, expected smoke-test duration/cost (a few cents), and get explicit go-ahead — per the user's instruction. The Stage 1→Stage 2 transition is a bigger decision now than originally scoped (launching N pods, sized from Stage 1's measured throughput, instead of continuing on the same single pod) — present the computed N, expected aggregate GPU-hours, and expected cost range before launching, as a lighter "proceeding unless you object" notice rather than a second hard gate, but with real numbers shown given the larger scale.

## Cost tracking & budget guardrails (`pipeline/costs.py`)

- `gpu_compute`: wall-clock pod uptime (`run_meta.pod_started_at`/`pod_stopped_at`) × the configured hourly rate (set from the actual booked rate, since Community Cloud pricing fluctuates) — simple and accurate per pod; each pod ledgers only its own uptime, summed across shards at merge time. Not fetched from a billing API (none exists at this granularity).
- `r2_storage` / `r2_class_a_ops` / `r2_class_b_ops`: computed from actual bytes uploaded and counted put/get/list calls against published R2 per-unit rates — at the 100+ clean hour scale, still likely (not assumed) inside R2's free tier (10GB + 1M Class A + 10M Class B ops/month); report the real measured number either way rather than assuming $0 by inspection.
- `egress`: explicitly logged as $0.00 (R2 has no egress fees) — stating "measured, zero by design" is more credible than silence, and the spec asks for this line item explicitly.
- Failed/wasted spend (e.g. a failed smoke test, a retried stage, a pod that stalls and gets terminated by the watchdog) gets its own ledger row so the final report's "wasted spend" line is real, not guessed.
- `pipeline_runner`'s main loop checks `running_total + projected_next_episode_cost` against `BUDGET_CAP_USD × 0.9` (and the analogous time check) before starting each new episode; if exceeded, that pod stops pulling new work, finishes in-flight work, exports, uploads, and self-stops. With N pods running, the per-pod cap is set to a fair share of the global $100 (with the watchdog tracking the true aggregate across all pods, since a pod can't see its siblings' spend on its own) — both are belt-and-suspenders given the cost numbers involved are still well under the cap even pessimistically.
- `scripts/merge_shards.py` concatenates every shard's `cost_events` ledger into one master ledger before `scripts/report.py` runs.
- `scripts/report.py` generates `COST_REPORT.md` straight from the merged ledger (cost per raw hour, cost per usable hour, category breakdown, per-shard breakdown) and includes a checklist for the manual evidence the spec wants for reimbursement: RunPod billing/usage export for every pod session, Cloudflare R2 usage screenshot, and an explicit reconciliation note comparing the ledger estimate to actual provider billing.
- This ledger and `COST_REPORT.md` stay scoped to infra costs only. The separate Claude/agent compute cost (see dedicated section above) is reported in `WRITEUP.md`, explicitly labeled as outside this budget/ledger.

## Staged rollout (time is the binding constraint, not money)

0. Network re-check (see prerequisite section) + register free accounts/keys (PodcastIndex, RunPod, R2 bucket, HF token + accept pyannote 3.1 + embedding model gated-model agreements).
1. **Local, zero-cost build & test** (Phase A — can start immediately): all of `pipeline/`, `scripts/`, `tests/`, Dockerfile, configs. Unit tests in `tests/` cover DB resumability, clip-segmentation math (synthetic VAD/diarization fixtures, no audio needed), quality-filter logic, manifest schema, cost-ledger math, and clustering math (synthetic embeddings) — all pure Python, no GPU, no network.
2. **Connectivity smoke test** (`scripts/smoke_test.py`): real PodcastIndex auth call, real R2 put/get/list of a throwaway object, real HF token fetch validating the gated-model agreement was accepted, real RunPod API call listing GPU types (no pod created). Free.
3. **Podcast selection** (`scripts/select_podcasts.py`): query PodcastIndex for enough interview/talk-style shows with consistent hosts to assemble a ~150–200 raw-hour candidate pool (expected ~10–20 shows; longer-episode shows preferred for better intro/outro/ad amortization), write `config/podcasts.json`. Free.
4. **Notify-before-spend checkpoint**, then **pod smoke test**: one short episode end-to-end on a real RTX 3090 pod; stop (don't terminate) afterward; manually inspect the resulting clip + manifest row + heartbeat.
5. **Stage 1 (validate + calibrate)**: single pod, ~2–3 podcasts × ~3–5 episodes, full pipeline, then two things happen off this one batch: (a) a deliberate **human checkpoint** — listen to a sample of clips, confirm speaker IDs look consistent across ≥2 episodes of the same podcast, tune `match_threshold`/duration-bucket ratios if needed (Libri-Light-style threshold-by-histogram-inspection, per the Prior art section); (b) **throughput calibration** — record measured wall-clock-per-raw-hour for the diarization stage (the expected bottleneck) on this real hardware/audio, replacing the planning-stage 1x–40x literature spread with one real number.
6. **Pod-count sizing checkpoint**: compute `pods_needed = ceil(remaining_raw_hours / (measured_hours_per_wallclock_hour × stage2_wallclock_budget))` from Stage 1's measurement, round up with a +20–30% safety margin, then partition all remaining `queued` episodes across that many shards (`scripts/partition_episodes.py`, bin-packed by reported duration). Present the computed pod count, expected aggregate GPU-hours, and expected cost range to the user as a lighter "proceeding unless you object" notice (not a second hard gate, since the smoke test already confirmed real spend is safe) before launching.
7. **Stage 2 (scale, multi-pod)**: launch the computed number of pods, each bound to its own shard via `--shard-id`, processing disjoint `assigned_shard` episodes independently — no cross-pod coordination needed during processing. `scripts/poll_status.py` aggregates heartbeats across all pods, watches for stale heartbeats or cap breaches, and surfaces (not auto-executes) a "launch more pods" recommendation if aggregate pace falls behind the 100+ clean-hour target. Stop queuing new work once the aggregate clean-hour count crosses 100+ with margin; let in-flight episodes finish.
8. **Merge + centralized re-cluster**: once all shards finish (or are stopped), `scripts/merge_shards.py` downloads each pod's `pipeline.db` from R2, merges `local_speaker_segments`/`clips`/`cost_events` (collision-free, globally-unique keys), then re-runs `recluster_podcast_from_scratch` once per podcast over the merged embeddings — discarding every shard-local provisional `speakers`/`clips.speaker_id` value in favor of one centrally-computed, globally-consistent answer. This is the step that makes multi-pod parallelism safe for speaker-ID accuracy; it runs even for podcasts that happened to stay within one shard, for uniform correctness.
9. **Report generation**: `scripts/report.py` → `PROCESSING_SUMMARY.md`, `COST_REPORT.md` (now per-shard-plus-aggregate); `scripts/validate_manifest.py` sanity-checks the final merged JSONL; `WRITEUP.md`/`LIMITATIONS.md` written up covering approach, throughput-vs-cost tradeoffs for every major decision (including the single-pod-vs-multi-pod sizing decision and its measured basis), prior-art grounding, speaker-ID methodology/failure modes (including the sharding hazard and its fix), resumability/monitoring/bottleneck analysis, at-scale (10,000+ hour) estimate with a Libri-Light-style tiered-output discussion, and the informational Claude/agent compute cost disclosure.

Target corpus (~150–200 raw hours across ~10–20 podcasts, netting 100+ clean hours at the assumed ~65% yield) no longer fits comfortably on one pod under a pessimistic RTF, which is exactly why pod count is now sized empirically from Stage 1's measurement rather than assumed — see the Scale revision section. Even a pessimistic 12 pods × 12h × $0.30–0.44/hr lands at ~$43–63, still under the $100 cap with room to spare; money remains the looser constraint than wall-clock time, but at this scale it's tracked deliberately rather than treated as a rounding error. The rollout still doesn't try to "use up" the $100 — the real risk to the actual optimization metric (100+ clean hours within the 24h window, at unchanged speaker-ID accuracy) is a late-stage failure forcing a restart or a pod-count guess that lands short, not underspending.

## Verification plan

- **No GPU, no network** (do this first, in this session): `tests/test_db.py` (state-machine resumability, including simulated mid-stage failure, rerun-skips-done-work, and `assigned_shard` filtering), `tests/test_segment.py` (synthetic VAD/diarization fixtures → expected clip durations/exclusions), `tests/test_quality.py`, `tests/test_manifest.py` (schema match to the spec's exact example), `tests/test_costs.py`, `tests/test_cluster.py` (synthetic embeddings with known cluster structure, including a test for `recluster_podcast_from_scratch` correctly ignoring shard-local provisional IDs and rebuilding consistent global ones from merged segments). Note: ffmpeg-dependent code can only be smoke-tested wherever ffmpeg is actually installed, which may not be this exact sandbox.
- **Network required, still ~free**: `scripts/smoke_test.py`, `scripts/select_podcasts.py` against the real PodcastIndex API.
- **Needs the real GPU pod**: real timing numbers for VAD/diarization/ASR on an actual RTX 3090 (this is the Stage 1 throughput calibration that the pod-count formula consumes directly — not just a budget sanity-check anymore), gated-model auth working end-to-end, the human-listening checkpoint in Stage 1 (speaker-ID sanity can't be fully automated), and — once Stage 2 runs — an end-to-end check that `merge_shards.py` + `recluster_podcast_from_scratch` produce one consistent speaker ID per real person for any podcast whose episodes landed across more than one shard.

## Deliverables mapping

`WRITEUP.md` (approach + the requested throughput-vs-cost rationale per decision, now including the single-pod-vs-multi-pod sizing tradeoff + Prior art subsection + informational Claude/agent compute cost disclosure), dataset link (R2 bucket/prefix, 100+ clean hours), this repo as the code deliverable, a sample manifest export, `PROCESSING_SUMMARY.md` (raw/usable hours, clip count, duration histogram, discard-reason counts, per-shard breakdown), `COST_REPORT.md` (trial-run breakdown + at-scale estimate, including the 10,000+ hour what-changes discussion: e.g. further pod/queue scaling, dedicated music/ad classifier, centroid-only re-clustering, batched multi-GPU ASR, and a Libri-Light-style tiered strict/broad output split), a dedicated speaker-ID section (methodology + failure modes above, explicitly naming the lack of precedent in LibriSpeech/Libri-Light and the sharding hazard + fix), and `LIMITATIONS.md` (diarization/ASR/speaker-consistency/overlap/ads-music-noise caveats, explicitly naming music/ad detection as the weakest link given no dedicated classifier is in the chosen stack, and naming cross-episode speaker re-identification — now sharpened by the multi-pod merge step — as the least-precedented component).
