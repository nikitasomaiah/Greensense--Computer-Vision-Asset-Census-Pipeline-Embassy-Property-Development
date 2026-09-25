"""
asset_register.py — GreenSense Tree & Asset Census module
-----------------------------------------------------------
Turns detection results into a running digital asset register.
Uses SQLite for robust storage while preserving backward compatibility with CSV
read/write interfaces.

Core SOP alignment:
  - "Checking the plants growth (Trees, palms, shrubs, ground covers and lawn)"
  - Daily Landscape Checklist (Annexure 1)
  - Half-yearly SME Landscaping Audit export

Core feature: LOSS/DAMAGE ALERTING.
If today's tree/asset count for a zone drops meaningfully below the recent rolling
average, a high-visibility loss alert is triggered.
"""

from __future__ import annotations

import csv
import os
import sqlite3
from datetime import datetime, date
from io import BytesIO, StringIO
from statistics import mean
from typing import Dict, List, Optional, Union

import pandas as pd

DB_PATH = "asset_register.db"
LEGACY_CSV_PATH = "asset_register.csv"
LOSS_ALERT_RATIO = 0.7  # 30% drop triggers an alert flag


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Returns a connection to the SQLite database, initializing tables if needed."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH, csv_path: str = LEGACY_CSV_PATH) -> None:
    """Initializes the SQLite database and auto-migrates legacy CSV records if present."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS census_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                zone TEXT NOT NULL,
                image_path TEXT NOT NULL,
                label TEXT NOT NULL,
                count INTEGER NOT NULL CHECK(count >= 0),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

        # Check if table is empty and legacy CSV exists for migration
        cursor.execute("SELECT COUNT(*) FROM census_log")
        row_count = cursor.fetchone()[0]

        if row_count == 0 and os.path.exists(csv_path):
            print(f"Migrating legacy CSV '{csv_path}' to SQLite database '{db_path}'...")
            try:
                with open(csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        try:
                            _validate_census_input(
                                zone=r["zone"],
                                counts={r["label"]: int(r["count"])},
                                census_date=r["date"],
                            )
                            cursor.execute(
                                """
                                INSERT INTO census_log (date, zone, image_path, label, count)
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                (r["date"], r["zone"], r["image_path"], r["label"], int(r["count"])),
                            )
                        except (ValueError, KeyError):
                            continue
                conn.commit()
                print("Legacy CSV migration completed successfully.")
            except Exception as e:
                print(f"Warning: Failed to migrate legacy CSV: {e}")


def _validate_census_input(zone: str, counts: Dict[str, int], census_date: str) -> None:
    """Validates parameters before logging to prevent register corruption."""
    if not isinstance(zone, str) or not zone.strip():
        raise ValueError("Zone name cannot be empty or non-string.")

    try:
        datetime.strptime(census_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        raise ValueError(f"Invalid date format '{census_date}'. Expected 'YYYY-MM-DD'.")

    if not isinstance(counts, dict):
        raise ValueError("Counts must be a dictionary of asset label to count.")

    for label, count in counts.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Asset label cannot be empty.")
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Count for '{label}' must be a non-negative integer, got: {count}")


def log_census(
    zone: str,
    counts: Dict[str, int],
    image_path: str,
    census_date: Optional[str] = None,
    register_path: str = DB_PATH,
) -> None:
    """
    Append one census result (one photo's worth of counts) to the register database.
    Rejects negative counts, empty zone names, or invalid date formats.
    """
    census_date = census_date or date.today().isoformat()
    _validate_census_input(zone=zone, counts=counts, census_date=census_date)

    # If register_path points to SQLite db (default)
    if register_path.endswith(".db"):
        init_db(db_path=register_path)
        with get_db_connection(register_path) as conn:
            cursor = conn.cursor()
            for label, count in counts.items():
                cursor.execute(
                    """
                    INSERT INTO census_log (date, zone, image_path, label, count)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (census_date, zone.strip(), image_path, label, count),
                )
            conn.commit()
    else:
        # Fallback to CSV write if caller explicitly passes a .csv path
        file_exists = os.path.exists(register_path)
        with open(register_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "zone", "image_path", "label", "count"])
            if not file_exists:
                writer.writeheader()
            for label, count in counts.items():
                writer.writerow(
                    {
                        "date": census_date,
                        "zone": zone.strip(),
                        "image_path": image_path,
                        "label": label,
                        "count": count,
                    }
                )


def load_register(register_path: str = DB_PATH) -> List[dict]:
    """Load all records from the register (SQLite database or CSV file)."""
    if register_path.endswith(".db"):
        if not os.path.exists(register_path):
            return []
        init_db(db_path=register_path)
        with get_db_connection(register_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT date, zone, image_path, label, count FROM census_log ORDER BY date ASC, id ASC"
            )
            rows = cursor.fetchall()
            return [
                {
                    "date": r["date"],
                    "zone": r["zone"],
                    "image_path": r["image_path"],
                    "label": r["label"],
                    "count": str(r["count"]),
                }
                for r in rows
            ]
    else:
        if not os.path.exists(register_path):
            return []
        with open(register_path, newline="") as f:
            return list(csv.DictReader(f))


def zone_history(zone: str, label: str, register_path: str = DB_PATH) -> List[dict]:
    """All historical counts for one zone + asset type, sorted by date."""
    target_zone = zone.strip() if isinstance(zone, str) else zone
    rows = [
        r for r in load_register(register_path)
        if r["zone"] == target_zone and r["label"] == label
    ]
    return sorted(rows, key=lambda r: r["date"])


def check_for_loss(
    zone: str,
    label: str,
    latest_count: int,
    register_path: str = DB_PATH,
    lookback: int = 5,
) -> Optional[str]:
    """
    Compare latest_count to the rolling average of the last `lookback`
    readings for this zone/label. Returns a human-readable alert string
    if a likely loss/damage event is detected, else None.
    """
    history = zone_history(zone, label, register_path)
    if len(history) < 2:
        return None  # Not enough historical baseline data yet

    recent = [int(r["count"]) for r in history[-lookback:]]
    baseline = mean(recent)

    if baseline == 0:
        return None

    if latest_count < baseline * LOSS_ALERT_RATIO:
        drop = baseline - latest_count
        return (
            f"⚠ Possible loss/damage in zone '{zone}': '{label}' count dropped "
            f"from an average of {baseline:.1f} to {latest_count} "
            f"(down {drop:.1f}). Recommend a site check and replacement order."
        )
    return None


def summary_report(register_path: str = DB_PATH) -> Dict[str, Dict[str, int]]:
    """
    Latest count per zone per asset type — the 'current state' snapshot,
    e.g. for a half-yearly SME audit export.
    """
    rows = load_register(register_path)
    latest: Dict[str, Dict[str, dict]] = {}
    for r in rows:
        zone, label = r["zone"], r["label"]
        latest.setdefault(zone, {})
        existing = latest[zone].get(label)
        if existing is None or r["date"] >= existing["date"]:
            latest[zone][label] = r

    return {
        zone: {label: int(rec["count"]) for label, rec in labels.items()}
        for zone, labels in latest.items()
    }


def get_known_zones(register_path: str = DB_PATH) -> List[str]:
    """Returns sorted list of distinct zone names present in the register."""
    rows = load_register(register_path)
    zones = sorted(list(set(r["zone"] for r in rows if r.get("zone"))))
    return zones


def export_audit_report(
    output_format: str = "csv", register_path: str = DB_PATH
) -> Union[str, bytes]:
    """
    Exports current summary audit report as CSV string or Excel bytes,
    formatted for the half-yearly SME Landscaping Audit.
    """
    report = summary_report(register_path)
    data = []
    for zone, labels in report.items():
        for label, count in labels.items():
            if count > 0:
                data.append({"Zone": zone, "Asset Category": label, "Latest Count": count})

    df = pd.DataFrame(data)
    if df.empty:
        df = pd.DataFrame(columns=["Zone", "Asset Category", "Latest Count"])

    if output_format.lower() == "excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Landscaping Audit")
        return output.getvalue()
    else:
        return df.to_csv(index=False)


def clear_test_data(register_path: str = DB_PATH) -> int:
    """Deletes all rows where zone starts with TEST_. Returns number of rows removed."""
    if register_path.endswith(".db"):
        if not os.path.exists(register_path):
            return 0
        init_db(db_path=register_path)
        with get_db_connection(register_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM census_log WHERE zone LIKE 'TEST_%'")
            removed = cursor.rowcount
            conn.commit()
            return removed
    else:
        if not os.path.exists(register_path):
            return 0
        rows = load_register(register_path)
        kept_rows = [r for r in rows if not r.get("zone", "").startswith("TEST_")]
        removed = len(rows) - len(kept_rows)
        if removed > 0:
            with open(register_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["date", "zone", "image_path", "label", "count"])
                writer.writeheader()
                writer.writerows(kept_rows)
        return removed

