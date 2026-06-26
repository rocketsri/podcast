"""Tests for pipeline/costs.py: cost arithmetic, free-tier netting, the
record_* writers (verified against db.total_cost), and budget/time soft-stop
threshold math. Pure arithmetic + in-memory sqlite, no network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline import costs, db


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    db.init_db(connection)
    yield connection
    connection.close()


# --- raw cost arithmetic -----------------------------------------------------------


def test_gpu_compute_cost_usd():
    assert costs.gpu_compute_cost_usd(2.0, 0.5) == 1.0
    assert costs.gpu_compute_cost_usd(0.0, 0.5) == 0.0


def test_r2_storage_cost_usd():
    assert costs.r2_storage_cost_usd(10.0) == pytest.approx(10.0 * 0.015)
    assert costs.r2_storage_cost_usd(0.0) == 0.0


def test_r2_class_a_cost_usd():
    assert costs.r2_class_a_cost_usd(1_000_000) == pytest.approx(4.50)
    assert costs.r2_class_a_cost_usd(500_000) == pytest.approx(2.25)
    assert costs.r2_class_a_cost_usd(0) == 0.0


def test_r2_class_b_cost_usd():
    assert costs.r2_class_b_cost_usd(1_000_000) == pytest.approx(0.36)
    assert costs.r2_class_b_cost_usd(10_000_000) == pytest.approx(3.60)
    assert costs.r2_class_b_cost_usd(0) == 0.0


def test_gb_months_for_bytes():
    one_gb = 1024 ** 3
    one_month_hours = 24 * 30
    assert costs.gb_months_for_bytes(one_gb, one_month_hours) == pytest.approx(1.0)
    assert costs.gb_months_for_bytes(one_gb, one_month_hours / 2) == pytest.approx(0.5)


# --- apply_free_tier ----------------------------------------------------------------


def test_apply_free_tier_zero_billable_when_under_allowance():
    result = costs.apply_free_tier(gb_months=5.0, class_a_ops=500_000, class_b_ops=5_000_000)
    assert result["storage_usd"] == 0.0
    assert result["class_a_usd"] == 0.0
    assert result["class_b_usd"] == 0.0
    assert result["within_free_tier"] is True


def test_apply_free_tier_exactly_at_allowance_is_free():
    result = costs.apply_free_tier(
        gb_months=costs.R2_FREE_TIER_GB_MONTHS,
        class_a_ops=costs.R2_FREE_TIER_CLASS_A_OPS,
        class_b_ops=costs.R2_FREE_TIER_CLASS_B_OPS,
    )
    assert result["storage_usd"] == 0.0
    assert result["class_a_usd"] == 0.0
    assert result["class_b_usd"] == 0.0
    assert result["within_free_tier"] is True


def test_apply_free_tier_correct_excess_when_over_allowance():
    gb_months = costs.R2_FREE_TIER_GB_MONTHS + 20.0       # 20 GB-months over
    class_a_ops = costs.R2_FREE_TIER_CLASS_A_OPS + 200_000  # 200k ops over
    class_b_ops = costs.R2_FREE_TIER_CLASS_B_OPS + 2_000_000  # 2M ops over

    result = costs.apply_free_tier(gb_months, class_a_ops, class_b_ops)

    assert result["storage_usd"] == pytest.approx(20.0 * costs.R2_STORAGE_USD_PER_GB_MONTH)
    assert result["class_a_usd"] == pytest.approx((200_000 / 1_000_000) * costs.R2_CLASS_A_USD_PER_MILLION_OPS)
    assert result["class_b_usd"] == pytest.approx((2_000_000 / 1_000_000) * costs.R2_CLASS_B_USD_PER_MILLION_OPS)
    assert result["within_free_tier"] is False


def test_apply_free_tier_only_one_dimension_over_allowance():
    """Storage over the allowance but ops within it -- within_free_tier
    must still be False (any dimension over trips it), and the other two
    dimensions' billable amounts must be exactly zero."""
    result = costs.apply_free_tier(
        gb_months=costs.R2_FREE_TIER_GB_MONTHS + 1.0,
        class_a_ops=0,
        class_b_ops=0,
    )
    assert result["storage_usd"] > 0.0
    assert result["class_a_usd"] == 0.0
    assert result["class_b_usd"] == 0.0
    assert result["within_free_tier"] is False


# --- record_* writers actually persist rows total_cost() can see -------------------


def test_record_gpu_compute_writes_row_visible_to_total_cost(conn):
    amount = costs.record_gpu_compute(conn, hours=2.0, hourly_rate_usd=0.3, related_episode_id="ep1")
    assert amount == pytest.approx(0.6)
    assert db.total_cost(conn) == pytest.approx(0.6)
    row = conn.execute("SELECT * FROM cost_events WHERE category = 'gpu_compute'").fetchone()
    assert row["related_episode_id"] == "ep1"
    assert row["amount_usd"] == pytest.approx(0.6)


def test_record_r2_storage_writes_row_visible_to_total_cost(conn):
    one_gb = 1024 ** 3
    amount = costs.record_r2_storage(conn, total_bytes=one_gb, hours_stored=24 * 30)
    assert amount == pytest.approx(costs.R2_STORAGE_USD_PER_GB_MONTH)
    assert db.total_cost(conn) == pytest.approx(costs.R2_STORAGE_USD_PER_GB_MONTH)


def test_record_r2_class_a_ops_writes_row_visible_to_total_cost(conn):
    amount = costs.record_r2_class_a_ops(conn, op_count=2_000_000)
    assert amount == pytest.approx(9.0)
    assert db.total_cost(conn) == pytest.approx(9.0)


def test_record_r2_class_b_ops_writes_row_visible_to_total_cost(conn):
    amount = costs.record_r2_class_b_ops(conn, op_count=5_000_000)
    assert amount == pytest.approx(1.8)
    assert db.total_cost(conn) == pytest.approx(1.8)


def test_record_egress_records_measured_zero(conn):
    costs.record_egress(conn, bytes_egress=123456)
    assert db.total_cost(conn) == 0.0
    row = conn.execute("SELECT * FROM cost_events WHERE category = 'egress'").fetchone()
    assert row["amount_usd"] == 0.0
    assert "zero" in row["description"]


def test_multiple_record_calls_accumulate_in_total_cost(conn):
    costs.record_gpu_compute(conn, hours=1.0, hourly_rate_usd=0.3)
    costs.record_r2_storage(conn, total_bytes=1024 ** 3, hours_stored=24 * 30)
    costs.record_r2_class_a_ops(conn, op_count=1_000_000)
    costs.record_r2_class_b_ops(conn, op_count=1_000_000)
    expected = 0.3 + costs.R2_STORAGE_USD_PER_GB_MONTH + 4.50 + 0.36
    assert db.total_cost(conn) == pytest.approx(expected)


def test_record_wasted_spend_is_tagged_and_summed_by_total_wasted_spend(conn):
    costs.record_wasted_spend(conn, "gpu_compute", 1.25, "stalled pod", related_episode_id="ep1")
    costs.record_gpu_compute(conn, hours=1.0, hourly_rate_usd=0.3)  # not wasted

    assert db.total_cost(conn) == pytest.approx(1.25 + 0.3)
    assert costs.total_wasted_spend(conn) == pytest.approx(1.25)


def test_total_wasted_spend_zero_when_nothing_wasted(conn):
    costs.record_gpu_compute(conn, hours=1.0, hourly_rate_usd=0.3)
    assert costs.total_wasted_spend(conn) == 0.0


# --- per_pod_budget_cap_usd ----------------------------------------------------------


def test_per_pod_budget_cap_usd_divides_evenly():
    assert costs.per_pod_budget_cap_usd(100.0, 4) == pytest.approx(25.0)


def test_per_pod_budget_cap_usd_floors_pod_count_at_one():
    assert costs.per_pod_budget_cap_usd(100.0, 0) == pytest.approx(100.0)
    assert costs.per_pod_budget_cap_usd(100.0, -3) == pytest.approx(100.0)


def test_per_pod_budget_cap_usd_single_pod():
    assert costs.per_pod_budget_cap_usd(50.0, 1) == pytest.approx(50.0)


# --- should_stop_for_budget / should_stop_for_time ------------------------------------


def make_cost_cfg(budget_cap_usd=100.0, budget_soft_stop_fraction=0.9):
    return SimpleNamespace(budget_cap_usd=budget_cap_usd, budget_soft_stop_fraction=budget_soft_stop_fraction)


def make_time_cfg(time_cap_hours=24.0, time_soft_stop_fraction=0.9):
    return SimpleNamespace(time_cap_hours=time_cap_hours, time_soft_stop_fraction=time_soft_stop_fraction)


def test_should_stop_for_budget_false_when_well_under_soft_cap(conn):
    cfg = make_cost_cfg(budget_cap_usd=100.0, budget_soft_stop_fraction=0.9)
    db.record_cost_event(conn, "gpu_compute", 10.0)
    # soft cap = 90.0; current 10 + projected 5 = 15, well under.
    assert costs.should_stop_for_budget(conn, projected_next_cost_usd=5.0, cfg=cfg) is False


def test_should_stop_for_budget_true_when_projected_total_exceeds_soft_cap(conn):
    cfg = make_cost_cfg(budget_cap_usd=100.0, budget_soft_stop_fraction=0.9)
    db.record_cost_event(conn, "gpu_compute", 85.0)
    # soft cap = 90.0; current 85 + projected 10 = 95 > 90.
    assert costs.should_stop_for_budget(conn, projected_next_cost_usd=10.0, cfg=cfg) is True


def test_should_stop_for_budget_boundary_exactly_at_soft_cap_is_false():
    """The check is strictly greater-than, so landing exactly on the soft
    cap should not trigger a stop."""
    conn = db.connect(":memory:")
    db.init_db(conn)
    cfg = make_cost_cfg(budget_cap_usd=100.0, budget_soft_stop_fraction=0.9)
    db.record_cost_event(conn, "gpu_compute", 80.0)
    # current 80 + projected 10 = 90 == soft cap exactly -> not stopped.
    assert costs.should_stop_for_budget(conn, projected_next_cost_usd=10.0, cfg=cfg) is False
    conn.close()


def test_should_stop_for_time_false_when_under_soft_cap():
    cfg = make_time_cfg(time_cap_hours=24.0, time_soft_stop_fraction=0.9)
    # soft cap = 21.6h; 10 + 1 = 11, well under.
    assert costs.should_stop_for_time(elapsed_hours=10.0, projected_next_hours=1.0, cfg=cfg) is False


def test_should_stop_for_time_true_when_projected_exceeds_soft_cap():
    cfg = make_time_cfg(time_cap_hours=24.0, time_soft_stop_fraction=0.9)
    # soft cap = 21.6h; 20 + 5 = 25 > 21.6.
    assert costs.should_stop_for_time(elapsed_hours=20.0, projected_next_hours=5.0, cfg=cfg) is True


def test_should_stop_for_time_boundary_exactly_at_soft_cap_is_false():
    cfg = make_time_cfg(time_cap_hours=10.0, time_soft_stop_fraction=0.5)
    # soft cap = 5.0h exactly; 3 + 2 = 5.0 -> not stopped (strictly greater-than).
    assert costs.should_stop_for_time(elapsed_hours=3.0, projected_next_hours=2.0, cfg=cfg) is False
