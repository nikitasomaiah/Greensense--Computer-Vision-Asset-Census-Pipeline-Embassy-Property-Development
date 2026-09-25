"""
tests/test_asset_register.py — Pytest suite for GreenSense asset register
-----------------------------------------------------------------------------
Tests SQLite data persistence, input validation, loss/damage alert threshold
logic, edge cases, summary aggregation, and audit export capabilities.
"""

import os
import tempfile
import pytest
from asset_register import (
    check_for_loss,
    clear_test_data,
    export_audit_report,
    get_known_zones,
    init_db,
    load_register,
    log_census,
    summary_report,
    zone_history,
)


@pytest.fixture
def temp_db():
    """Provides a temporary SQLite database file for isolated test execution."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(db_path=path, csv_path="")
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_log_census_and_load_register(temp_db):
    """Test normal census logging and record retrieval."""
    counts = {"a tree": 5, "a palm tree": 2}
    log_census("Test Zone A", counts, "test1.jpg", census_date="2026-07-01", register_path=temp_db)

    records = load_register(register_path=temp_db)
    assert len(records) == 2
    tree_rec = next(r for r in records if r["label"] == "a tree")
    assert tree_rec["zone"] == "Test Zone A"
    assert tree_rec["count"] == "5"
    assert tree_rec["date"] == "2026-07-01"


def test_data_validation_negative_counts(temp_db):
    """Test that negative asset counts are rejected with a ValueError."""
    with pytest.raises(ValueError, match="Count for 'a tree' must be a non-negative integer"):
        log_census("Test Zone", {"a tree": -3}, "test.jpg", census_date="2026-07-01", register_path=temp_db)


def test_data_validation_empty_zone(temp_db):
    """Test that empty or whitespace-only zone names are rejected."""
    with pytest.raises(ValueError, match="Zone name cannot be empty"):
        log_census("   ", {"a tree": 5}, "test.jpg", census_date="2026-07-01", register_path=temp_db)


def test_data_validation_malformed_date(temp_db):
    """Test that invalid date formats are rejected."""
    with pytest.raises(ValueError, match="Invalid date format"):
        log_census("Test Zone", {"a tree": 5}, "test.jpg", census_date="01-07-2026", register_path=temp_db)


def test_loss_alert_first_ever_reading(temp_db):
    """Test that the first-ever reading for a zone does NOT trigger a loss alert (insufficient history)."""
    log_census("Zone B", {"a tree": 10}, "img1.jpg", census_date="2026-07-01", register_path=temp_db)
    alert = check_for_loss("Zone B", "a tree", latest_count=4, register_path=temp_db)
    assert alert is None


def test_loss_alert_zero_baseline(temp_db):
    """Test that zero baseline history does not cause division-by-zero or false alerts."""
    log_census("Zone B", {"a tree": 0}, "img1.jpg", census_date="2026-07-01", register_path=temp_db)
    log_census("Zone B", {"a tree": 0}, "img2.jpg", census_date="2026-07-02", register_path=temp_db)
    alert = check_for_loss("Zone B", "a tree", latest_count=0, register_path=temp_db)
    assert alert is None


def test_loss_alert_threshold_boundary(temp_db):
    """
    Test loss alert threshold boundary.
    Baseline = 10. LOSS_ALERT_RATIO = 0.7 (30% drop).
    Count = 8 (20% drop) -> No alert
    Count = 7 (30% drop) -> No alert (at boundary: 7.0 is not < 7.0)
    Count = 6 (40% drop) -> Alert triggered!
    """
    log_census("Zone C", {"a tree": 10}, "img1.jpg", census_date="2026-07-01", register_path=temp_db)
    log_census("Zone C", {"a tree": 10}, "img2.jpg", census_date="2026-07-02", register_path=temp_db)

    # 20% drop -> No alert
    assert check_for_loss("Zone C", "a tree", latest_count=8, register_path=temp_db) is None

    # 30% drop (exactly at 0.7 ratio) -> No alert
    assert check_for_loss("Zone C", "a tree", latest_count=7, register_path=temp_db) is None

    # 40% drop (< 0.7 ratio) -> Alert triggered!
    alert = check_for_loss("Zone C", "a tree", latest_count=6, register_path=temp_db)
    assert alert is not None
    assert "Possible loss/damage in zone 'Zone C'" in alert
    assert "dropped from an average of 10.0 to 6" in alert


def test_summary_report_aggregation(temp_db):
    """Test that summary_report aggregates to the latest date reading per zone and label."""
    log_census("Zone D", {"a tree": 10, "a hedge": 4}, "img1.jpg", census_date="2026-07-01", register_path=temp_db)
    log_census("Zone D", {"a tree": 12, "a hedge": 4}, "img2.jpg", census_date="2026-07-08", register_path=temp_db)
    log_census("Zone E", {"a palm tree": 3}, "img3.jpg", census_date="2026-07-05", register_path=temp_db)

    summary = summary_report(register_path=temp_db)
    assert "Zone D" in summary
    assert "Zone E" in summary
    assert summary["Zone D"]["a tree"] == 12  # Latest count from 2026-07-08
    assert summary["Zone D"]["a hedge"] == 4
    assert summary["Zone E"]["a palm tree"] == 3


def test_known_zones_retrieval(temp_db):
    """Test retrieving list of distinct known zones."""
    log_census("Zone Alpha", {"a tree": 1}, "img1.jpg", census_date="2026-07-01", register_path=temp_db)
    log_census("Zone Beta", {"a tree": 2}, "img2.jpg", census_date="2026-07-02", register_path=temp_db)
    zones = get_known_zones(register_path=temp_db)
    assert zones == ["Zone Alpha", "Zone Beta"]


def test_export_audit_report(temp_db):
    """Test CSV and Excel export functions."""
    log_census("Zone F", {"a tree": 8}, "img1.jpg", census_date="2026-07-01", register_path=temp_db)

    csv_out = export_audit_report(output_format="csv", register_path=temp_db)
    assert isinstance(csv_out, str)
    assert "Zone F" in csv_out
    assert "a tree" in csv_out
    assert "8" in csv_out

    excel_out = export_audit_report(output_format="excel", register_path=temp_db)
    assert isinstance(excel_out, bytes)
    assert len(excel_out) > 0


def test_zone_isolation_summary_and_history(temp_db):
    """Test that logging entries under two different zone names never mixes counts between them."""
    log_census("Manyata Block D Garden", {"a tree": 3}, "IMG_4788.jpg", census_date="2026-07-01", register_path=temp_db)
    log_census("Boulevard West Lawn", {"a tree": 20}, "boulevard.jpg", census_date="2026-07-01", register_path=temp_db)

    summary = summary_report(register_path=temp_db)
    assert "Manyata Block D Garden" in summary
    assert "Boulevard West Lawn" in summary
    assert summary["Manyata Block D Garden"]["a tree"] == 3
    assert summary["Boulevard West Lawn"]["a tree"] == 20

    history_manyata = zone_history("Manyata Block D Garden", "a tree", register_path=temp_db)
    history_boulevard = zone_history("Boulevard West Lawn", "a tree", register_path=temp_db)

    assert len(history_manyata) == 1
    assert history_manyata[0]["image_path"] == "IMG_4788.jpg"
    assert history_manyata[0]["count"] == "3"

    assert len(history_boulevard) == 1
    assert history_boulevard[0]["image_path"] == "boulevard.jpg"
    assert history_boulevard[0]["count"] == "20"


def test_clear_test_data(temp_db):
    """Test clearing test mode data logged with TEST_ prefix."""
    log_census("TEST_Manyata", {"a tree": 5}, "img1.jpg", census_date="2026-07-01", register_path=temp_db)
    log_census("Real Zone", {"a tree": 10}, "img2.jpg", census_date="2026-07-01", register_path=temp_db)

    assert len(load_register(register_path=temp_db)) == 2
    removed = clear_test_data(register_path=temp_db)
    assert removed == 1

    records = load_register(register_path=temp_db)
    assert len(records) == 1
    assert records[0]["zone"] == "Real Zone"


