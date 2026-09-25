"""
generate_sample_register.py — GreenSense Data Pipeline Generator
------------------------------------------------------------------
1. Generates synthetic landscape images under sample_data/ with embedded EXIF GPS tags.
2. Pre-populates historical asset census records across multiple dates and zones
   in the asset register (asset_register.db / asset_register.csv).
3. Simulates a tree loss event in 'Manyata Block D Garden' to demonstrate the
   loss/damage alerting feature on dashboard startup.
"""

import os
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from asset_register import log_census, init_db
from utils.geotag import extract_gps_info, KNOWN_ZONE_COORDINATES

SAMPLE_DIR = "sample_data"

# Sample zones and baseline counts
SAMPLES = [
    {
        "zone": "Manyata Block D Garden",
        "filename": "manyata_block_d.jpg",
        "lat": 13.0455,
        "lon": 77.6201,
        "baseline": {
            "a tree": 12,
            "a palm tree": 6,
            "a flowering shrub": 18,
            "a hedge": 8,
            "a lawn / grass area": 2,
        },
        "simulated_drop": {  # Simulates loss on latest date
            "a tree": 7,  # Drop from 12 to 7 (41% drop -> triggers alert!)
        },
    },
    {
        "zone": "Embassy GolfLinks Zone 1",
        "filename": "golflinks_zone_1.jpg",
        "lat": 12.9602,
        "lon": 77.6484,
        "baseline": {
            "a tree": 15,
            "a palm tree": 10,
            "a flowering shrub": 25,
            "a hedge": 12,
            "a garden pathway": 3,
        },
    },
    {
        "zone": "Boulevard West Lawn",
        "filename": "boulevard_west_lawn.jpg",
        "lat": 13.1605,
        "lon": 77.5802,
        "baseline": {
            "a tree": 20,
            "a palm tree": 4,
            "a flowering shrub": 30,
            "a lawn / grass area": 4,
            "a pergola": 1,
        },
    },
]


def _create_exif_bytes(lat: float, lon: float) -> Image.Exif:
    """Construct PIL Exif object containing GPS metadata."""
    exif = Image.Exif()
    # Convert decimal lat/lon to deg, min, sec rationals
    def to_dms(val):
        d = int(val)
        m = int((val - d) * 60)
        s = round((val - d - m / 60.0) * 3600.0, 2)
        return (d, m, s)

    lat_dms = to_dms(abs(lat))
    lon_dms = to_dms(abs(lon))

    gps_ifd = {
        1: "N" if lat >= 0 else "S",
        2: lat_dms,
        3: "E" if lon >= 0 else "W",
        4: lon_dms,
    }
    exif[0x8825] = gps_ifd  # 0x8825 is GPSInfo tag ID
    return exif


def create_synthetic_landscape_image(filepath: str, title: str, lat: float, lon: float):
    """Draw a synthetic garden/landscape scene and save with EXIF GPS metadata."""
    width, height = 800, 600
    img = Image.new("RGB", (width, height), color=(135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)

    # Draw grass ground
    draw.rectangle([0, 300, width, height], fill=(34, 139, 34))  # Forest green

    # Draw garden pathway
    draw.polygon([(350, 600), (450, 600), (420, 380), (380, 380)], fill=(210, 180, 140))

    # Draw synthetic trees
    tree_positions = [(100, 280), (250, 260), (600, 290), (720, 270)]
    for x, y in tree_positions:
        draw.rectangle([x - 15, y, x + 15, y + 80], fill=(101, 67, 33))  # Trunk
        draw.ellipse([x - 50, y - 90, x + 50, y + 10], fill=(0, 100, 0))  # Canopy

    # Draw synthetic palm trees
    palm_positions = [(450, 250), (520, 240)]
    for x, y in palm_positions:
        draw.rectangle([x - 10, y, x + 10, y + 120], fill=(139, 69, 19))
        draw.line([x, y, x - 40, y - 30], fill=(46, 139, 87), width=8)
        draw.line([x, y, x + 40, y - 30], fill=(46, 139, 87), width=8)
        draw.line([x, y, x - 50, y], fill=(46, 139, 87), width=8)
        draw.line([x, y, x + 50, y], fill=(46, 139, 87), width=8)

    # Draw shrubs / hedges
    shrub_positions = [(80, 420), (200, 450), (620, 430), (700, 460)]
    for x, y in shrub_positions:
        draw.ellipse([x - 30, y - 20, x + 30, y + 20], fill=(50, 205, 50))

    # Draw text label overlay
    draw.rectangle([20, 20, 400, 70], fill=(0, 0, 0, 128))
    draw.text((30, 30), f"GreenSense Fixture: {title}", fill=(255, 255, 255))
    draw.text((30, 50), f"GPS: {lat:.4f} N, {lon:.4f} E", fill=(200, 255, 200))

    exif = _create_exif_bytes(lat, lon)
    img.save(filepath, "JPEG", exif=exif)
    print(f"Generated synthetic image with EXIF GPS: {filepath}")


def generate_all():
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    init_db()  # Ensure database table exists

    print("Step 1: Generating sample landscape images with embedded GPS EXIF tags...")
    for item in SAMPLES:
        path = os.path.join(SAMPLE_DIR, item["filename"])
        create_synthetic_landscape_image(path, item["zone"], item["lat"], item["lon"])

    print("\nStep 2: Populating historical asset register records over simulated dates...")
    today = datetime.now().date()
    dates = [today - timedelta(days=7 * i) for i in range(5, -1, -1)]  # 6 weeks of data

    for item in SAMPLES:
        zone = item["zone"]
        filename = os.path.join(SAMPLE_DIR, item["filename"])
        baseline = item["baseline"]
        simulated_drop = item.get("simulated_drop", {})

        for idx, census_date in enumerate(dates):
            date_str = census_date.isoformat()
            is_latest = (idx == len(dates) - 1)

            counts = {}
            for label, base_cnt in baseline.items():
                if is_latest and label in simulated_drop:
                    counts[label] = simulated_drop[label]
                else:
                    # Slight natural variation (+/- 0 or 1) to simulate realistic counting noise
                    noise = int(np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2]))
                    counts[label] = max(0, base_cnt + noise)

            log_census(zone=zone, counts=counts, image_path=filename, census_date=date_str)
            print(f"Logged census for '{zone}' on {date_str}: {counts}")

    print("\nSample data generation complete! SQLite register pre-populated successfully.")


if __name__ == "__main__":
    generate_all()
