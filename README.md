# Podcast Speech Dataset Pipeline

Converts public podcast audio into a LibriLight-style speech dataset: short, clean, single-speaker FLAC clips with persistent cross-episode speaker IDs and a full JSONL manifest. Built for the DataOps take-home (`FILE_3000.pdf`).

**Status at submission**: 102.40 usable hours delivered (693 episodes, 104,983 clean clips, 58 podcasts) for $16.02 of infra spend — 16% of the $100 budget — with 6 RunPod pods actively converging on roughly 385 usable hours within the next few hours. All numbers below are a live snapshot; rerun `scripts/report.py` against a fresh merge for the current state.

## Deliverables index

| # | Deliverable | File |
| --- | --- | --- |
| 1 | Approach/tradeoffs writeup | [`WRITEUP.md`](WRITEUP.md) |
| 2 | Processed dataset output | R2 bucket, see below |
| 3 | Processing code | this repo |
| 4 | Sample metadata manifest | [`manifest_sample.jsonl`](manifest_sample.jsonl) (60 rows, `random.seed(42)` sample of the live 104,983-row manifest) |
| 5 | Processing summary (raw/usable hours, clip count, yield, duration distribution, discard reasons) | [`PROCESSING_SUMMARY.md`](PROCESSING_SUMMARY.md) |
| 6 | Trial-run cost breakdown | [`COST_REPORT.md`](COST_REPORT.md) |
| 7 | At-scale cost estimate | [`COST_REPORT.md`](COST_REPORT.md) (At-scale estimate section) |
| 8 | Speaker ID / cross-episode matching explanation | [`WRITEUP.md`](WRITEUP.md) (Speaker ID section) |
| 9 | Resumability, failure handling, monitoring, bottleneck notes | [`WRITEUP.md`](WRITEUP.md) (Resumability/monitoring section) |
| 10 | Known limitations | [`LIMITATIONS.md`](LIMITATIONS.md) |

Background design document (research, prior-art grounding, full architecture rationale, written before/during the build): [`PLAN.md`](PLAN.md). Chronological log of real problems hit and fixed during the live run: [`PROBLEMS.md`](PROBLEMS.md).

## Processed dataset output (R2)

Cloudflare R2, bucket `podcast`, all objects namespaced under key prefix `v2/`. The bucket's S3 API remains private (R2 credentials required, see Setup below), but every object is also reachable read-only over plain HTTPS via a public Cloudflare Worker:

**Public base URL**: `https://podcast-dataset-public.podcast-dataset-rocketsri.workers.dev/<key>`

e.g. the live manifest is `https://podcast-dataset-public.podcast-dataset-rocketsri.workers.dev/v2/manifest/manifest.jsonl`, and a clip is `.../v2/clips/<podcast_id>/<episode_id>/<clip_id>.flac`. GET/HEAD only, supports byte-range requests (for streaming/seeking audio clients) and CORS (`access-control-allow-origin: *`). Worker source: `infra/r2_public_worker/worker.js`; deploy config: `infra/r2_public_worker/wrangler.toml`.

**Browsing**: any path ending in `/` (including the bare base URL) renders an HTML directory listing instead of fetching an object — open `https://podcast-dataset-public.podcast-dataset-rocketsri.workers.dev/` in a browser and click through `v2/` → `clips/` → a podcast → an episode to find real clip files, or jump straight to `v2/manifest/manifest.jsonl` for the full metadata. Listings paginate via an R2 cursor (`?cursor=...` link at the bottom) when a directory's underlying key count exceeds R2's per-call scan budget — this can require several "next page" clicks at the `v2/clips/` level since some podcasts have thousands of clips nested under one prefix.

This exposes the **whole bucket**, not just the dataset-relevant prefixes below — a deliberate scope choice (simplest, matches the standard R2-public-read Worker pattern) over building custom per-prefix access control. None of the exposed objects contain secrets (credentials only ever lived in `.env`/RunPod env injection, never uploaded), but `db_snapshots/`/`logs/`/`status/` are internal operational artifacts, not part of the intended dataset deliverable — readers should treat `clips/` + `manifest/` as the actual data product.

| Content | Key pattern |
| --- | --- |
| Clip audio (FLAC, 16kHz mono) | `v2/clips/<podcast_id>/<episode_id>/<clip_id>.flac` |
| Manifest (JSONL, one row per clean clip) | `v2/manifest/manifest.jsonl` |
| Per-pod database snapshots | `v2/db_snapshots/<pod_id>/pipeline.db` |
| Per-pod heartbeat/status | `v2/status/<pod_id>.json` |
| Per-pod logs | `v2/logs/<pod_id>.log` |

`v2/manifest/manifest.jsonl` is refreshed by `scripts/merge_shards.py` (the only thing that ever writes that key — live pods never touch it) and reflects the live merge of all 6 shards' current database state at submission time.

## Repo layout

```
pipeline/        # every pipeline stage as an independent module (see module docstrings)
config/          # pipeline.yaml (all tunables) + podcasts.json (selected shows)
scripts/         # discovery, smoke test, partitioning, merge, report, validation, backfill
infra/           # bootstrap.sh -- the actual RunPod pod entrypoint
tests/           # pytest suite, no GPU/network required
run_pipeline.py  # CLI entrypoint (what infra/bootstrap.sh invokes on a real pod)
```

## Setup / rerun

1. `pip install -r requirements.txt` (pinned versions; on a real GPU box, `pip uninstall -y torchvision` afterward — see `infra/bootstrap.sh` and `PROBLEMS.md` for the torch/torchvision conflict this works around).
2. Copy `.env.example` to `.env` and fill in: `PODCASTINDEX_API_KEY`/`SECRET` (free account at podcastindex.org), `RUNPOD_API_KEY`, `R2_ACCOUNT_ID`/`ACCESS_KEY_ID`/`SECRET_ACCESS_KEY`/`BUCKET_NAME` (Cloudflare R2), `HF_TOKEN` (with the pyannote 3.1 + embedding model gated-model agreements accepted on huggingface.co), `BUDGET_CAP_USD`/`TIME_CAP_HOURS` guardrails.
3. Local/dev run (no GPU, no R2 required): `python3 run_pipeline_local.py` — uses a CPU diarization fallback (`pipeline/local_diarize.py`) for iteration without real GPU access.
4. Real GPU pod run: `python3 run_pipeline.py --db work/pipeline.db --work-dir work --log-path work/pipeline.log --pod-id <pod-id> --device cuda` — this is exactly what `infra/bootstrap.sh` runs on a RunPod pod; pass `--shard <n>` for a multi-pod fleet (set up via `python3 scripts/partition_episodes.py --shards <n>` first, run once against the shared queue before any pod starts), `--max-episodes` to cap a smoke test, `--no-upload` to skip R2 (clips stay local).
5. After a fleet run: `python3 scripts/merge_shards.py --auto-discover --output-db work/merged.db --manifest-out work/manifest.jsonl` to merge every pod's R2-snapshotted database and re-run cross-episode speaker clustering from scratch per podcast; `python3 scripts/report.py --db work/merged.db --out-dir .` to regenerate `PROCESSING_SUMMARY.md`/`COST_REPORT.md`; `python3 scripts/validate_manifest.py --manifest work/manifest.jsonl` to sanity-check schema/duration bounds.
6. Tests: `python3 -m pytest tests/` — pure Python, no GPU/network required.

## Verifying the live submission without any credentials

Everything below works from a clean checkout with no `.env`, no R2/RunPod keys, and no GPU — it pulls straight from the public Worker URL above:

```
curl -o manifest.jsonl https://podcast-dataset-public.podcast-dataset-rocketsri.workers.dev/v2/manifest/manifest.jsonl
python3 scripts/validate_manifest.py --manifest manifest.jsonl   # schema + duration-bound checks against the real, live manifest
head -1 manifest.jsonl   # grab a clip_id/podcast_id/episode_id to build a clip URL, e.g.:
curl -o sample_clip.flac https://podcast-dataset-public.podcast-dataset-rocketsri.workers.dev/v2/clips/<podcast_id>/<episode_id>/<clip_id>.flac
```

That last `curl` pulls a real 16kHz mono FLAC clip end-to-end — enough to confirm the manifest and the audio agree, with zero infra setup.

## Reading order

Start with `WRITEUP.md` for the approach and the speaker-ID/bottleneck discussion, then `LIMITATIONS.md` for what's genuinely uncertain or imperfect about the live output, then `PROCESSING_SUMMARY.md`/`COST_REPORT.md` for the numbers backing both. `PLAN.md` has the full original design rationale (prior-art research, algorithm details, staged-rollout plan) for anyone who wants the "why" behind a specific module beyond what's summarized in `WRITEUP.md`.
