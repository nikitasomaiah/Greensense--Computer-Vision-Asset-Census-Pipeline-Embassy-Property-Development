"""
utils/geotag.py — GreenSense EXIF GPS Metadata Reader
------------------------------------------------------
Extracts embedded EXIF GPS latitude and longitude coordinates from site photos.
Maps GPS coordinates to known Embassy facility zones when possible, with graceful
fallback to manual zone entry when EXIF metadata is missing or corrupted.
"""

from typing import Dict, Optional, Tuple, Union
from PIL import Image, ExifTags


# Pre-defined reference GPS coordinates for Embassy facility zones in Bengaluru
KNOWN_ZONE_COORDINATES = [
    {
        "zone": "Manyata Block D Garden",
        "lat": 13.0455,
        "lon": 77.6201,
        "radius_km": 2.0,
    },
    {
        "zone": "Embassy GolfLinks Zone 1",
        "lat": 12.9602,
        "lon": 77.6484,
        "radius_km": 2.0,
    },
    {
        "zone": "Boulevard West Lawn",
        "lat": 13.1605,
        "lon": 77.5802,
        "radius_km": 2.0,
    },
]


def _convert_to_degrees(value) -> float:
    """Helper function to convert EXIF GPS DMS (degrees, minutes, seconds) to decimal degrees."""
    try:
        if isinstance(value, (int, float)):
            return float(value)

        # Handle tuple of rationals/numbers: (deg, min, sec)
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])

        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        return 0.0


def extract_gps_info(image_input: Union[str, Image.Image]) -> Optional[Tuple[float, float]]:
    """
    Extract (latitude, longitude) from an image file path or PIL Image instance.

    Returns:
        (latitude, longitude) as float tuple in decimal degrees, or None if EXIF GPS data is absent or corrupt.
    """
    try:
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = image_input

        exif = image._getexif() if hasattr(image, "_getexif") else None
        if not exif:
            return None

        gps_info = {}
        for tag_id, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
            if tag_name == "GPSInfo":
                for gps_tag_id in value:
                    gps_tag_name = ExifTags.GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag_name] = value[gps_tag_id]

        if not gps_info:
            return None

        lat_data = gps_info.get("GPSLatitude")
        lat_ref = gps_info.get("GPSLatitudeRef", "N")
        lon_data = gps_info.get("GPSLongitude")
        lon_ref = gps_info.get("GPSLongitudeRef", "E")

        if not lat_data or not lon_data:
            return None

        lat = _convert_to_degrees(lat_data)
        if lat_ref != "N":
            lat = -lat

        lon = _convert_to_degrees(lon_data)
        if lon_ref != "E":
            lon = -lon

        if lat == 0.0 and lon == 0.0:
            return None

        return round(lat, 6), round(lon, 6)

    except Exception:
        # Fallback gracefully to None without interrupting the pipeline
        return None


def get_zone_from_gps(lat: float, lon: float) -> Optional[str]:
    """
    Finds the closest known Embassy zone within radius based on GPS coordinates.
    Returns zone name or None if no matching zone found within radius.
    """
    for entry in KNOWN_ZONE_COORDINATES:
        # Approximate Euclidean distance in degrees (~111km per degree)
        d_lat = abs(entry["lat"] - lat)
        d_lon = abs(entry["lon"] - lon)
        dist_approx_km = ((d_lat * 111.0) ** 2 + (d_lon * 111.0) ** 2) ** 0.5
        if dist_approx_km <= entry["radius_km"]:
            return entry["zone"]
    return None


def extract_photo_metadata(image_input: Union[str, Image.Image]) -> Dict[str, Optional[Union[Tuple[float, float], str]]]:
    """
    Extracts high-level photo metadata including GPS coordinates and inferred zone name.
    """
    coords = extract_gps_info(image_input)
    zone = get_zone_from_gps(coords[0], coords[1]) if coords else None
    return {
        "gps": coords,
        "suggested_zone": zone,
    }
