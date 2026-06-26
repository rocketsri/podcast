"""SQLite -> PROCESSING_SUMMARY.md, COST_REPORT.md, per the plan's
Deliverables-mapping + Cost-tracking sections.

Reads an already-merged pipeline.db (post scripts/merge_shards.py -- this
script does not merge shards itself, see plan's Scale revision section) and
writes two markdown reports into --out-dir: PROCESSING_SUMMARY.md (raw/usable
hours, yield, stage/discard breakdowns, duration-distribution histogram vs
config target ratios, per-podcast breakdown) and COST_REPORT.md (total cost,
category breakdown, cost per raw/usable hour, wasted spend, a manual
reimbursement-evidence checklist).

Run with: python3 scripts/report.py --db work/pipeline.db [--config path] [--out-dir .]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import config, costs, db, logging_utils  # noqa: E402

logger = logging_utils.get_logger()

SECONDS_PER_HOUR = 3600.0

# Matches config/pipeline.yaml's segmentation.target_bucket_ratios keys ->
# (lower_bound_inclusive, upper_bound_exclusive] duration_seconds bucket.
DURATION_BUCKETS = (
    ("under_10s", 0.0, 10.0),
    ("from_10_to_20s", 10.0, 20.0),
    ("from_20_to_30s", 20.0, 30.0),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH), help="path to the (already-merged) pipeline.db")
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    parser.add_argument("--out-dir", default=".", help="directory to write PROCESSING_SUMMARY.md / COST_REPORT.md into")
    return parser.parse_args(argv)


# --- formatting helpers -------------------------------------------------------

def _fmt_hours(seconds: float) -> str:
    return f"{(seconds or 0.0) / SECONDS_PER_HOUR:.2f}h"


def _fmt_pct(fraction: float) -> str:
    return f"{fraction * 100:.1f}%"


def _fmt_usd(amount: float) -> str:
    return f"${amount:,.2f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_no data_\n"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


# --- data gathering ------------------------------------------------------------

def _episode_totals(conn) -> dict:
    row = conn.execute(
        "SELECT COALESCE(SUM(raw_seconds), 0) AS raw_total,"
        " COALESCE(SUM(usable_seconds), 0) AS usable_total,"
        " COUNT(*) AS total_episodes"
        " FROM episodes"
    ).fetchone()
    return {"raw_seconds": row["raw_total"], "usable_seconds": row["usable_total"], "total_episodes": row["total_episodes"]}


def _stage_counts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT stage, COUNT(*) AS n FROM episodes GROUP BY stage ORDER BY n DESC").fetchall()


def _failed_stage_breakdown(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT COALESCE(failed_stage, '(unknown)') AS failed_stage, COUNT(*) AS n"
        " FROM episodes WHERE stage = ? GROUP BY failed_stage ORDER BY n DESC",
        (db.FAILED_STAGE,),
    ).fetchall()


def _clip_totals(conn) -> dict:
    row = conn.execute(
        "SELECT COUNT(*) AS total,"
        " SUM(CASE WHEN uploaded = 1 THEN 1 ELSE 0 END) AS uploaded,"
        " SUM(CASE WHEN discard_reason IS NOT NULL THEN 1 ELSE 0 END) AS discarded"
        " FROM clips"
    ).fetchone()
    return {
        "total": row["total"] or 0,
        "uploaded": row["uploaded"] or 0,
        "discarded": row["discarded"] or 0,
    }


def _discard_histogram(conn) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT discard_reason, COUNT(*) AS n FROM clips"
        " WHERE discard_reason IS NOT NULL GROUP BY discard_reason ORDER BY n DESC"
    ).fetchall()


def _duration_bucket_counts(conn) -> tuple[dict[str, int], int]:
    """Bucket every clip's duration_seconds into DURATION_BUCKETS. Clips
    outside [0, 30s] (should not happen given segment.py's hard 30s cap, but
    a merged/multi-shard db is exactly the kind of place a quiet upstream bug
    would first show up) are counted separately, not silently dropped."""
    counts = {name: 0 for name, _, _ in DURATION_BUCKETS}
    overflow = 0
    rows = conn.execute("SELECT duration_seconds FROM clips").fetchall()
    for row in rows:
        dur = row["duration_seconds"]
        placed = False
        for name, lo, hi in DURATION_BUCKETS:
            if lo <= dur < hi or (name == DURATION_BUCKETS[-1][0] and dur == hi):
                counts[name] += 1
                placed = True
                break
        if not placed:
            overflow += 1
    return counts, overflow


def _per_podcast_breakdown(conn) -> list[dict]:
    podcasts = conn.execute("SELECT podcast_id, title FROM podcasts ORDER BY podcast_id").fetchall()
    out = []
    for pod in podcasts:
        pod_id = pod["podcast_id"]
        ep_row = conn.execute(
            "SELECT COUNT(*) AS total,"
            " SUM(CASE WHEN stage = 'done' THEN 1 ELSE 0 END) AS done,"
            " COALESCE(SUM(usable_seconds), 0) AS usable_seconds"
            " FROM episodes WHERE podcast_id = ?",
            (pod_id,),
        ).fetchone()
        clip_row = conn.execute(
            "SELECT COUNT(*) AS total FROM clips WHERE podcast_id = ?", (pod_id,)
        ).fetchone()
        speaker_row = conn.execute(
            "SELECT COUNT(DISTINCT speaker_id) AS n FROM clips WHERE podcast_id = ? AND speaker_id IS NOT NULL",
            (pod_id,),
        ).fetchone()
        out.append(
            {
                "podcast_id": pod_id,
                "title": pod["title"],
                "episodes_total": ep_row["total"] or 0,
                "episodes_done": ep_row["done"] or 0,
                "usable_seconds": ep_row["usable_seconds"] or 0.0,
                "clips_total": clip_row["total"] or 0,
                "distinct_speakers": speaker_row["n"] or 0,
            }
        )
    return out


def _cost_category_breakdown(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT category, COALESCE(SUM(amount_usd), 0) AS total FROM cost_events"
        " GROUP BY category ORDER BY total DESC"
    ).fetchall()


def _has_pod_attribution(conn) -> bool:
    """cost_events has no pod_id column in the schema (pipeline/db.py) --
    per-pod attribution would require merge_shards.py to have tagged rows
    with which shard/pod produced them during merge. Check run_meta for any
    sign that happened rather than assuming either way."""
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(cost_events)")]
    if "pod_id" in cols:
        return True
    # Fallback: metadata_json might carry a pod/shard tag even without a
    # dedicated column -- sample a few rows rather than scanning the whole
    # table for a report that should stay cheap.
    rows = conn.execute(
        "SELECT metadata_json FROM cost_events WHERE metadata_json IS NOT NULL LIMIT 50"
    ).fetchall()

    for row in rows:
        try:
            meta = json.loads(row["metadata_json"])
        except (TypeError, ValueError):
            continue
        if isinstance(meta, dict) and ("pod_id" in meta or "shard_id" in meta):
            return True
    return False


# --- PROCESSING_SUMMARY.md ----------------------------------------------------

def build_processing_summary(conn: sqlite3.Connection, cfg, db_path_label: str) -> str:
    totals = _episode_totals(conn)
    raw_seconds = totals["raw_seconds"]
    usable_seconds = totals["usable_seconds"]
    yield_fraction = (usable_seconds / raw_seconds) if raw_seconds else 0.0

    stage_rows = _stage_counts(conn)
    failed_rows = _failed_stage_breakdown(conn)
    clip_totals = _clip_totals(conn)
    discard_rows = _discard_histogram(conn)
    bucket_counts, overflow = _duration_bucket_counts(conn)
    per_podcast = _per_podcast_breakdown(conn)

    lines: list[str] = []
    lines.append("# Processing Summary")
    lines.append("")
    lines.append(f"_Generated by `scripts/report.py` from `{db_path_label}`._")
    lines.append("")

    lines.append("## Headline numbers")
    lines.append("")
    lines.append(_md_table(
        ["Metric", "Value"],
        [
            ["Total episodes", str(totals["total_episodes"])],
            ["Total raw hours", _fmt_hours(raw_seconds)],
            ["Total usable/clean hours", _fmt_hours(usable_seconds)],
            ["Yield fraction (usable / raw)", _fmt_pct(yield_fraction)],
            ["Total clips produced", str(clip_totals["total"])],
            ["Clips uploaded", str(clip_totals["uploaded"])],
            ["Clips discarded", str(clip_totals["discarded"])],
        ],
    ))

    lines.append("## Episodes by stage")
    lines.append("")
    stage_table_rows = [[row["stage"], str(row["n"])] for row in stage_rows]
    lines.append(_md_table(["Stage", "Episode count"], stage_table_rows))

    failed_count = next((row["n"] for row in stage_rows if row["stage"] == db.FAILED_STAGE), 0)
    lines.append(f"**Failed episodes: {failed_count}**")
    lines.append("")
    if failed_rows:
        lines.append(_md_table(
            ["Failed at stage", "Count"],
            [[row["failed_stage"], str(row["n"])] for row in failed_rows],
        ))
    else:
        lines.append("_No failed episodes._")
        lines.append("")

    lines.append("## Clip discard-reason histogram")
    lines.append("")
    if discard_rows:
        total_discarded = clip_totals["discarded"] or 1
        lines.append(_md_table(
            ["Discard reason", "Count", "% of discarded clips"],
            [
                [row["discard_reason"], str(row["n"]), _fmt_pct(row["n"] / total_discarded)]
                for row in discard_rows
            ],
        ))
    else:
        lines.append("_No discarded clips._")
        lines.append("")

    lines.append("## Clip duration-distribution vs configured target ratios")
    lines.append("")
    target_ratios = cfg.segmentation.target_bucket_ratios.as_dict()
    bucketed_total = sum(bucket_counts.values())
    dist_rows = []
    for name, lo, hi in DURATION_BUCKETS:
        actual_count = bucket_counts[name]
        actual_frac = (actual_count / bucketed_total) if bucketed_total else 0.0
        target_frac = target_ratios.get(name, 0.0)
        delta_pp = (actual_frac - target_frac) * 100
        label = f"{name} ({lo:g}-{hi:g}s)"
        dist_rows.append([
            label, str(actual_count), _fmt_pct(actual_frac), _fmt_pct(target_frac),
            f"{delta_pp:+.1f}pp",
        ])
    lines.append(_md_table(["Bucket", "Count", "Actual %", "Target %", "Delta"], dist_rows))
    if overflow:
        lines.append(
            f"**WARNING: {overflow} clip(s) had a duration outside the expected "
            f"[0, {DURATION_BUCKETS[-1][2]:g}]s range covered by these buckets** -- "
            "investigate as a possible segmentation bug (segment.py is supposed to "
            "hard-cap clips at the configured max_clip_duration_seconds)."
        )
        lines.append("")

    lines.append("## Per-podcast breakdown")
    lines.append("")
    if per_podcast:
        lines.append(_md_table(
            ["Podcast", "Episodes done / total", "Clips", "Usable hours", "Distinct speakers"],
            [
                [
                    f"{p['title']} (`{p['podcast_id']}`)",
                    f"{p['episodes_done']} / {p['episodes_total']}",
                    str(p["clips_total"]),
                    _fmt_hours(p["usable_seconds"]),
                    str(p["distinct_speakers"]),
                ]
                for p in per_podcast
            ],
        ))
    else:
        lines.append("_No podcasts in the database._")
        lines.append("")

    return "\n".join(lines) + "\n"


# --- COST_REPORT.md -----------------------------------------------------------

def build_cost_report(conn: sqlite3.Connection, cfg, db_path_label: str) -> str:
    totals = _episode_totals(conn)
    raw_seconds = totals["raw_seconds"]
    usable_seconds = totals["usable_seconds"]
    total = db.total_cost(conn)
    wasted = costs.total_wasted_spend(conn)
    category_rows = _cost_category_breakdown(conn)

    cost_per_raw_hour = (total / (raw_seconds / SECONDS_PER_HOUR)) if raw_seconds else None
    cost_per_usable_hour = (total / (usable_seconds / SECONDS_PER_HOUR)) if usable_seconds else None

    lines: list[str] = []
    lines.append("# Cost Report")
    lines.append("")
    lines.append(f"_Generated by `scripts/report.py` from `{db_path_label}`._")
    lines.append("")
    lines.append(
        "Scoped to infra costs only (RunPod GPU compute + Cloudflare R2), per "
        "`pipeline/costs.py`'s ledger scope. The separate Claude/agent compute cost "
        "is informational and lives in `WRITEUP.md`, not here."
    )
    lines.append("")

    lines.append("## Headline numbers")
    lines.append("")
    lines.append(_md_table(
        ["Metric", "Value"],
        [
            ["Total cost (all cost_events)", _fmt_usd(total)],
            ["Total wasted spend", _fmt_usd(wasted)],
            ["Cost per raw hour", _fmt_usd(cost_per_raw_hour) if cost_per_raw_hour is not None else "n/a (no raw_seconds recorded)"],
            ["Cost per usable hour", _fmt_usd(cost_per_usable_hour) if cost_per_usable_hour is not None else "n/a (no usable_seconds recorded)"],
            ["Budget cap (config)", _fmt_usd(cfg.cost.budget_cap_usd)],
            ["% of budget cap spent", _fmt_pct(total / cfg.cost.budget_cap_usd) if cfg.cost.budget_cap_usd else "n/a"],
        ],
    ))

    lines.append("## Cost by category")
    lines.append("")
    if category_rows:
        lines.append(_md_table(
            ["Category", "Total USD", "% of total"],
            [
                [row["category"], _fmt_usd(row["total"]), _fmt_pct(row["total"] / total) if total else "n/a"]
                for row in category_rows
            ],
        ))
    else:
        lines.append("_No cost_events recorded._")
        lines.append("")

    lines.append("## Wasted spend")
    lines.append("")
    lines.append(
        f"**{_fmt_usd(wasted)}** of the total above is tagged `wasted` in "
        "`cost_events.metadata_json` (failed smoke tests, retried stages, pods "
        "stopped by the watchdog -- see `pipeline/costs.py:record_wasted_spend`)."
    )
    lines.append("")

    lines.append("## Per-pod cost attribution")
    lines.append("")
    if _has_pod_attribution(conn):
        pod_rows = conn.execute(
            "SELECT COALESCE(json_extract(metadata_json, '$.pod_id'), json_extract(metadata_json, '$.shard_id'), 'unknown') AS pod,"
            " COALESCE(SUM(amount_usd), 0) AS total"
            " FROM cost_events GROUP BY pod ORDER BY total DESC"
        ).fetchall()
        lines.append(_md_table(
            ["Pod / shard", "Total USD"],
            [[str(row["pod"]), _fmt_usd(row["total"])] for row in pod_rows],
        ))
    else:
        lines.append(
            "`cost_events` has no `pod_id` column and no `pod_id`/`shard_id` key in "
            "`metadata_json` for this db -- per-pod cost attribution requires "
            "`scripts/merge_shards.py` to have tagged rows with their source pod/shard "
            "during the merge. That tagging was not found in this db, so only the "
            "aggregate total above is reported rather than guessing a per-pod split."
        )
        lines.append("")

    lines.append("## Manual reimbursement-evidence checklist")
    lines.append("")
    lines.append(
        "The items below are explicitly meant to be filled in by a human after "
        "this report is generated, not computed from the db (per the plan's Cost "
        "tracking section) -- they reconcile this ledger against actual provider "
        "billing."
    )
    lines.append("")
    lines.append("- [ ] RunPod billing/usage export attached for every pod session")
    lines.append("- [ ] Cloudflare R2 usage dashboard screenshot attached")
    lines.append(
        "- [ ] Reconciliation note written comparing this ledger's total "
        f"({_fmt_usd(total)}) to actual provider billing, explaining any delta"
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = config.load_config(args.config)

    conn = db.connect(args.db)
    try:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_md = build_processing_summary(conn, cfg, args.db)
        cost_md = build_cost_report(conn, cfg, args.db)

        summary_path = out_dir / "PROCESSING_SUMMARY.md"
        cost_path = out_dir / "COST_REPORT.md"
        summary_path.write_text(summary_md)
        cost_path.write_text(cost_md)

        logger.info("wrote %s", summary_path)
        logger.info("wrote %s", cost_path)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
