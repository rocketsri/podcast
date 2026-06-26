"""Validates a JSONL dataset manifest against pipeline/manifest.py's schema,
plus sanity checks the plan's Deliverables-mapping section asks for: clip
count, duration-distribution vs configured target ratios, null-speaker_id
count, distinct speaker_id count per podcast, and out-of-bound durations.

This is a cheap pass/fail check in the same spirit as scripts/poll_status.py:
exit 0 means clean (zero schema errors, zero out-of-bound durations), exit 1
means something needs a human look. Distribution-ratio deviation is flagged
loudly in the printed report but does NOT alone fail the exit code, since a
skewed-but-otherwise-valid manifest is a quality signal to act on, not a
structural defect that should block downstream consumption.

Run with: python3 scripts/validate_manifest.py --manifest path/to/manifest.jsonl [--config path]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import config, logging_utils, manifest  # noqa: E402

logger = logging_utils.get_logger()

# How many percentage points an actual duration-bucket fraction may deviate
# from the configured target before being flagged. Chosen as a round,
# clearly-stated threshold per the task's "use your judgement and state it"
# instruction -- tight enough to catch a real segmentation regression,
# loose enough not to flag normal episode-to-episode content variance.
RATIO_DEVIATION_THRESHOLD_PP = 10.0

DURATION_BUCKETS = (
    ("under_10s", 0.0, 10.0),
    ("from_10_to_20s", 10.0, 20.0),
    ("from_20_to_30s", 20.0, 30.0),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="path to the JSONL manifest file")
    parser.add_argument("--config", default=None, help="path to pipeline.yaml (for segmentation bounds/target ratios)")
    return parser.parse_args(argv)


def _bucket_for(duration: float) -> str | None:
    for name, lo, hi in DURATION_BUCKETS:
        if lo <= duration < hi or (name == DURATION_BUCKETS[-1][0] and duration == hi):
            return name
    return None


def validate_manifest_file(path: Path, cfg) -> tuple[bool, list[str]]:
    """Returns (ok, report_lines). `ok` is False iff there are schema errors
    or out-of-bound durations -- the two conditions that mean the manifest
    itself is structurally broken, not just statistically skewed."""
    min_dur = cfg.segmentation.min_clip_duration_seconds
    max_dur = cfg.segmentation.max_clip_duration_seconds
    target_ratios = cfg.segmentation.target_bucket_ratios.as_dict()

    schema_errors: list[tuple[int, str]] = []
    parse_errors: list[tuple[int, str]] = []
    out_of_bounds: list[tuple[int, str, float]] = []

    clip_count = 0
    null_speaker_count = 0
    speakers_by_podcast: dict[str, set[str]] = defaultdict(set)
    bucket_counts: Counter[str] = Counter()
    unbucketed_count = 0

    with path.open() as f:
        for line_no, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                parse_errors.append((line_no, str(exc)))
                continue

            errors = manifest.validate_manifest_row(row)
            if errors:
                for err in errors:
                    schema_errors.append((line_no, err))
                continue  # don't trust shape-invalid rows for the stats below

            clip_count += 1

            speaker_id = row["speaker_id"]
            if speaker_id is None:
                null_speaker_count += 1
            else:
                speakers_by_podcast[row["podcast_id"]].add(speaker_id)

            duration = row["duration_seconds"]
            if duration < min_dur or duration > max_dur:
                out_of_bounds.append((line_no, row["clip_id"], duration))

            bucket = _bucket_for(duration)
            if bucket is None:
                unbucketed_count += 1
            else:
                bucket_counts[bucket] += 1

    lines: list[str] = []
    lines.append(f"Manifest: {path}")
    lines.append(f"Clip count (structurally valid rows): {clip_count}")
    lines.append("")

    # --- JSON parse errors ---
    if parse_errors:
        lines.append(f"JSON PARSE ERRORS: {len(parse_errors)}")
        for line_no, err in parse_errors[:20]:
            lines.append(f"  line {line_no}: {err}")
        if len(parse_errors) > 20:
            lines.append(f"  ... and {len(parse_errors) - 20} more")
        lines.append("")

    # --- schema errors ---
    if schema_errors:
        lines.append(f"SCHEMA ERRORS: {len(schema_errors)}")
        for line_no, err in schema_errors[:50]:
            lines.append(f"  line {line_no}: {err}")
        if len(schema_errors) > 50:
            lines.append(f"  ... and {len(schema_errors) - 50} more")
        lines.append("")
    else:
        lines.append("Schema errors: 0 (all rows match pipeline.manifest.validate_manifest_row)")
        lines.append("")

    # --- duration distribution vs target ---
    lines.append("Duration-distribution histogram vs configured target ratios "
                  f"(flag threshold: >{RATIO_DEVIATION_THRESHOLD_PP:.0f}pp deviation):")
    bucketed_total = sum(bucket_counts.values())
    any_ratio_flag = False
    for name, lo, hi in DURATION_BUCKETS:
        count = bucket_counts[name]
        actual_frac = (count / bucketed_total) if bucketed_total else 0.0
        target_frac = target_ratios.get(name, 0.0)
        delta_pp = (actual_frac - target_frac) * 100
        flag = abs(delta_pp) > RATIO_DEVIATION_THRESHOLD_PP
        any_ratio_flag = any_ratio_flag or flag
        marker = "  <-- FLAGGED: deviates from target by more than threshold" if flag else ""
        lines.append(
            f"  {name} ({lo:g}-{hi:g}s): {count} clips, actual={actual_frac * 100:.1f}% "
            f"target={target_frac * 100:.1f}% delta={delta_pp:+.1f}pp{marker}"
        )
    if unbucketed_count:
        lines.append(f"  (unbucketed -- duration outside [0, {DURATION_BUCKETS[-1][2]:g}]s): {unbucketed_count}")
    lines.append("")

    # --- null speaker_id ---
    lines.append(f"Clips with null speaker_id: {null_speaker_count} (of {clip_count}, "
                 f"{(null_speaker_count / clip_count * 100) if clip_count else 0:.1f}%)")
    lines.append("")

    # --- distinct speakers per podcast ---
    lines.append("Distinct speaker_id count per podcast_id:")
    if speakers_by_podcast:
        for podcast_id in sorted(speakers_by_podcast):
            lines.append(f"  {podcast_id}: {len(speakers_by_podcast[podcast_id])} distinct speakers")
    else:
        lines.append("  (no non-null speaker_id values found)")
    lines.append("")

    # --- out-of-bound durations: loud correctness flag, not a quiet stat ---
    if out_of_bounds:
        lines.append(
            f"*** CORRECTNESS BUG: {len(out_of_bounds)} clip(s) have duration_seconds outside "
            f"the configured bound [{min_dur:g}, {max_dur:g}]s. This should never happen -- "
            "segment.py is supposed to enforce both the min_clip_duration_seconds floor and "
            "the max_clip_duration_seconds hard cap before a clip ever reaches the manifest. "
            "Investigate upstream (pipeline/segment.py) rather than treating this as expected "
            "variance. ***"
        )
        for line_no, clip_id, duration in out_of_bounds[:20]:
            lines.append(f"  line {line_no}: clip_id={clip_id} duration_seconds={duration}")
        if len(out_of_bounds) > 20:
            lines.append(f"  ... and {len(out_of_bounds) - 20} more")
        lines.append("")
    else:
        lines.append(f"Out-of-bound durations (outside [{min_dur:g}, {max_dur:g}]s): 0")
        lines.append("")

    if any_ratio_flag:
        lines.append(
            f"NOTE: duration-distribution ratio deviated from target by more than "
            f"{RATIO_DEVIATION_THRESHOLD_PP:.0f}pp in at least one bucket (see flagged "
            "bucket(s) above). This is a quality signal worth investigating "
            "(segment.py's bucket-biasing heuristic, or real content skew), but does "
            "not by itself fail this check's exit code."
        )
        lines.append("")

    ok = not parse_errors and not schema_errors and not out_of_bounds
    lines.append("OK -- manifest is structurally valid and all durations in bounds" if ok
                 else "FAIL -- see PARSE/SCHEMA ERRORS or CORRECTNESS BUG sections above")
    return ok, lines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("manifest file not found: %s", manifest_path)
        return 1

    cfg = config.load_config(args.config)
    ok, lines = validate_manifest_file(manifest_path, cfg)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
