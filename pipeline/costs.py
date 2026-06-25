"""Cost ledger + budget guardrails: a thin domain layer over db.py's
append-only cost_events table (gpu_compute|r2_storage|r2_class_a_ops|
r2_class_b_ops|egress|other). Scoped to infra costs only, per the spec's
actual reimbursement ask -- the separate informational Claude/agent-compute
cost figure lives in WRITEUP.md, never in this ledger.

R2 per-unit rates and free-tier allowances below are Cloudflare's published
numbers as of this build (recalled from training knowledge -- this sandbox's
egress proxy blocks cloudflare.com, so they can't be live-verified here);
reconcile against actual R2 billing for the final COST_REPORT.md, per the
spec's own reimbursement-verification ask.
"""

from __future__ import annotations

import json
import sqlite3

from pipeline import db

GPU_HOURLY_USD_DEFAULT = 0.30  # overwritten by the actual booked RunPod rate at run start

R2_STORAGE_USD_PER_GB_MONTH = 0.015
R2_CLASS_A_USD_PER_MILLION_OPS = 4.50   # writes/lists (PutObject, ListObjectsV2, ...)
R2_CLASS_B_USD_PER_MILLION_OPS = 0.36   # reads (GetObject, HeadObject, ...)
R2_FREE_TIER_GB_MONTHS = 10.0
R2_FREE_TIER_CLASS_A_OPS = 1_000_000
R2_FREE_TIER_CLASS_B_OPS = 10_000_000
BYTES_PER_GB = 1024 ** 3


def gpu_compute_cost_usd(hours: float, hourly_rate_usd: float) -> float:
    return hours * hourly_rate_usd


def r2_storage_cost_usd(gb_months: float) -> float:
    return gb_months * R2_STORAGE_USD_PER_GB_MONTH


def r2_class_a_cost_usd(op_count: int) -> float:
    return (op_count / 1_000_000) * R2_CLASS_A_USD_PER_MILLION_OPS


def r2_class_b_cost_usd(op_count: int) -> float:
    return (op_count / 1_000_000) * R2_CLASS_B_USD_PER_MILLION_OPS


def gb_months_for_bytes(total_bytes: float, hours_stored: float) -> float:
    """Fractional GB-months a given byte count represents over `hours_stored`
    -- R2 bills storage per GB-month, but a multi-hour trial run only
    occupies a small fraction of a billing month."""
    gb = total_bytes / BYTES_PER_GB
    months = hours_stored / (24 * 30)
    return gb * months


def apply_free_tier(gb_months: float, class_a_ops: int, class_b_ops: int) -> dict:
    """Nets R2's free-tier allowance against measured usage -- used once at
    report time (scripts/report.py), over run-wide totals, not per-event,
    since the free tier is a monthly account-wide allowance, not something a
    single cost_events row can reason about in isolation."""
    billable_gb_months = max(0.0, gb_months - R2_FREE_TIER_GB_MONTHS)
    billable_class_a = max(0, class_a_ops - R2_FREE_TIER_CLASS_A_OPS)
    billable_class_b = max(0, class_b_ops - R2_FREE_TIER_CLASS_B_OPS)
    return {
        "storage_usd": r2_storage_cost_usd(billable_gb_months),
        "class_a_usd": r2_class_a_cost_usd(billable_class_a),
        "class_b_usd": r2_class_b_cost_usd(billable_class_b),
        "within_free_tier": billable_gb_months == 0.0 and billable_class_a == 0 and billable_class_b == 0,
    }


def record_gpu_compute(
    conn: sqlite3.Connection, hours: float, hourly_rate_usd: float,
    description: str = "", related_episode_id: str | None = None,
) -> float:
    amount = gpu_compute_cost_usd(hours, hourly_rate_usd)
    db.record_cost_event(
        conn, "gpu_compute", amount,
        description=description or f"{hours:.3f}h @ ${hourly_rate_usd:.4f}/hr",
        related_episode_id=related_episode_id,
        metadata={"hours": hours, "hourly_rate_usd": hourly_rate_usd},
    )
    return amount


def record_r2_storage(conn: sqlite3.Connection, total_bytes: float, hours_stored: float, description: str = "") -> float:
    gb_months = gb_months_for_bytes(total_bytes, hours_stored)
    amount = r2_storage_cost_usd(gb_months)
    db.record_cost_event(
        conn, "r2_storage", amount,
        description=description or f"{total_bytes / BYTES_PER_GB:.4f}GB for {hours_stored:.2f}h",
        metadata={"total_bytes": total_bytes, "hours_stored": hours_stored, "gb_months": gb_months},
    )
    return amount


def record_r2_class_a_ops(conn: sqlite3.Connection, op_count: int, description: str = "") -> float:
    amount = r2_class_a_cost_usd(op_count)
    db.record_cost_event(
        conn, "r2_class_a_ops", amount, description=description or f"{op_count} class A ops",
        metadata={"op_count": op_count},
    )
    return amount


def record_r2_class_b_ops(conn: sqlite3.Connection, op_count: int, description: str = "") -> float:
    amount = r2_class_b_cost_usd(op_count)
    db.record_cost_event(
        conn, "r2_class_b_ops", amount, description=description or f"{op_count} class B ops",
        metadata={"op_count": op_count},
    )
    return amount


def record_egress(conn: sqlite3.Connection, bytes_egress: int = 0) -> None:
    """R2 has no egress fees -- logged explicitly as measured-zero rather
    than omitted, since the spec asks for this line item by name and
    "measured, zero by design" is more credible than silence."""
    db.record_cost_event(
        conn, "egress", 0.0,
        description="R2 egress is free; measured zero by design",
        metadata={"bytes_egress": bytes_egress},
    )


def record_wasted_spend(
    conn: sqlite3.Connection, category: str, amount_usd: float, description: str,
    related_episode_id: str | None = None,
) -> None:
    """A failed smoke test, a retried stage, or a pod stalled-and-terminated
    by the watchdog -- tagged via metadata_json (the schema has no dedicated
    column for this) so report.py's "wasted spend" line is computed from real
    ledger rows, not guessed after the fact."""
    db.record_cost_event(
        conn, category, amount_usd, description=description,
        related_episode_id=related_episode_id, metadata={"wasted": True},
    )


def total_wasted_spend(conn: sqlite3.Connection) -> float:
    rows = conn.execute("SELECT amount_usd, metadata_json FROM cost_events WHERE metadata_json IS NOT NULL").fetchall()
    total = 0.0
    for row in rows:
        metadata = json.loads(row["metadata_json"])
        if metadata.get("wasted"):
            total += row["amount_usd"]
    return total


def per_pod_budget_cap_usd(global_cap_usd: float, pod_count: int) -> float:
    """A fair share of the global cap for one pod in a multi-pod Stage 2 run
    -- belt-and-suspenders alongside the watchdog's aggregate tracking, since
    no single pod can see its siblings' spend on its own."""
    return global_cap_usd / max(pod_count, 1)


def should_stop_for_budget(conn: sqlite3.Connection, projected_next_cost_usd: float, cfg) -> bool:
    projected_total = db.total_cost(conn) + projected_next_cost_usd
    return projected_total > cfg.budget_cap_usd * cfg.budget_soft_stop_fraction


def should_stop_for_time(elapsed_hours: float, projected_next_hours: float, cfg) -> bool:
    return (elapsed_hours + projected_next_hours) > cfg.time_cap_hours * cfg.time_soft_stop_fraction
