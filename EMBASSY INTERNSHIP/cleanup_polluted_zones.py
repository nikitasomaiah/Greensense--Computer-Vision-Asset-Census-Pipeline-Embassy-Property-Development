"""
cleanup_polluted_zones.py — Inspection Utility for Historical Zone Register
-----------------------------------------------------------------------------
Lists all zones currently in the SQLite asset register (`asset_register.db`)
along with their total record counts, unique image sources, and date ranges.

Allows manual review of historical test data logged prior to Test Mode isolation,
without auto-deleting any records.
"""

import sqlite3
import os

DB_PATH = "asset_register.db"


def inspect_register(db_path: str = DB_PATH):
    if not os.path.exists(db_path):
        print(f"Database file '{db_path}' not found.")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM census_log")
    total_records = cursor.fetchone()[0]

    print("=========================================================================")
    print(f"               GreenSense Asset Register Inspection Report              ")
    print("=========================================================================")
    print(f"Database File:  {db_path}")
    print(f"Total Entries:  {total_records} logged asset rows\n")

    cursor.execute("""
        SELECT 
            zone, 
            COUNT(*) as record_count,
            COUNT(DISTINCT image_path) as photo_count,
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM census_log 
        GROUP BY zone 
        ORDER BY zone ASC
    """)

    zones = cursor.fetchall()
    if not zones:
        print("No zone entries found in the database.")
        return

    print(f"{'Zone Name':<35} | {'Row Count':<10} | {'Photos':<8} | {'Date Range'}")
    print("-" * 75)
    for z in zones:
        date_range = f"{z['min_date']} to {z['max_date']}"
        print(f"{z['zone']:<35} | {z['record_count']:<10} | {z['photo_count']:<8} | {date_range}")

    print("-" * 75)
    print("\n[NOTE] Review the zones above for any test runs logged under production zone names.")
    print("To remove specific test rows, run targeted SQL queries or use clear_test_data() for TEST_ prefixed zones.\n")

    conn.close()


if __name__ == "__main__":
    inspect_register()
