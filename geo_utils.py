"""
Utility functions for geo-reasoning evaluation.
Includes haversine distance, country/city lookups, and coordinate validation.
"""

import math
import re
from typing import Dict, Tuple, Optional, List
import country_converter as coco


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
    
    Returns:
        Distance in kilometers
    """
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate if coordinates are within valid ranges.
    
    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)
    
    Returns:
        True if valid, False otherwise
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180


def parse_coordinates_from_text(text: str) -> Optional[Tuple[float, float]]:
    """
    Extract latitude and longitude from various text formats.
    
    Supports formats:
        - "43.1278, -89.3898"
        - "Latitude: 43.1278, Longitude: -89.3898"
        - "Coordinates: 43.1278, -89.3898"
        - "(43.1278, -89.3898)"
    
    Args:
        text: Text containing coordinates
    
    Returns:
        (latitude, longitude) tuple or None if not found
    """
    # Pattern 1: "Coordinates: X, Y" or "Coords: X, Y"
    pattern1 = r'(?:Coordinates?|Coords?):\s*([-\d.]+)\s*,\s*([-\d.]+)'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if validate_coordinates(lat, lon):
            return (lat, lon)
    
    # Pattern 2: "Latitude: X, Longitude: Y"
    pattern2 = r'Latitude:\s*([-\d.]+).*?Longitude:\s*([-\d.]+)'
    match = re.search(pattern2, text, re.IGNORECASE | re.DOTALL)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if validate_coordinates(lat, lon):
            return (lat, lon)
    
    # Pattern 3: Simple "X, Y" (at end of text)
    pattern3 = r'([-\d.]+)\s*,\s*([-\d.]+)\s*$'
    match = re.search(pattern3, text.strip())
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if validate_coordinates(lat, lon):
            return (lat, lon)
    
    # Pattern 4: "(X, Y)"
    pattern4 = r'\(([-\d.]+)\s*,\s*([-\d.]+)\)'
    match = re.search(pattern4, text)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if validate_coordinates(lat, lon):
            return (lat, lon)
    
    return None


def normalize_country_name(country: str) -> str:
    """
    Normalize country name for comparison.
    
    Args:
        country: Country name
    
    Returns:
        Normalized country name
    """
    # Common normalizations
    normalizations = {
        "usa": "united states",
        "us": "united states",
        "united states of america": "united states",
        "uk": "united kingdom",
        "uae": "united arab emirates",
    }
    
    country_lower = country.lower().strip()
    
    # Remove "the" prefix
    country_lower = re.sub(r'^the\s+', '', country_lower)
    
    # Apply specific normalizations
    return normalizations.get(country_lower, country_lower)


def countries_match(pred_country: str, gt_country: str, fuzzy: bool = True) -> bool:
    """
    Check if two country names match.
    
    Args:
        pred_country: Predicted country name
        gt_country: Ground truth country name
        fuzzy: Allow fuzzy matching (substring, normalization)
    
    Returns:
        True if match, False otherwise
    """
    if not pred_country or not gt_country:
        return False
    
    pred_norm = normalize_country_name(pred_country)
    gt_norm = normalize_country_name(gt_country)
    
    # Exact match
    if pred_norm == gt_norm:
        return True
    
    # Fuzzy match
    if fuzzy:
        # Check if one is substring of other
        if pred_norm in gt_norm or gt_norm in pred_norm:
            return True
    
    return False


def extract_location_info(text: str) -> Dict:
    """
    Extract all location information from text.
    
    Args:
        text: Text containing location information
    
    Returns:
        dict with country, city, continent, latitude, longitude
    """
    result = {
        "country": None,
        "city": None,
        "continent": None,
        "latitude": None,
        "longitude": None
    }
    
    # Extract country
    country_match = re.search(r'Country:\s*([^,\n]+)', text, re.IGNORECASE)
    if country_match:
        result["country"] = country_match.group(1).strip()
    
    # Extract city
    city_match = re.search(r'City:\s*([^,\n]+)', text, re.IGNORECASE)
    if city_match:
        result["city"] = city_match.group(1).strip()
    
    # Extract continent
    continent_match = re.search(r'Continent:\s*([^,\n]+)', text, re.IGNORECASE)
    if continent_match:
        result["continent"] = continent_match.group(1).strip()
    
    # Extract coordinates
    coords = parse_coordinates_from_text(text)
    if coords:
        result["latitude"], result["longitude"] = coords
    
    return result


def calculate_distance_accuracy(distance_km: float) -> Dict[str, bool]:
    """
    Calculate accuracy at standard distance thresholds.
    
    Args:
        distance_km: Distance in kilometers
    
    Returns:
        dict with boolean flags for each threshold
    """
    thresholds = [1, 25, 200, 750, 2500]
    return {f"within_{t}km": distance_km <= t for t in thresholds}


# Country to continent mapping (subset, extend as needed)
COUNTRY_TO_CONTINENT = {
    "united states": "north america",
    "canada": "north america",
    "mexico": "north america",
    "brazil": "south america",
    "argentina": "south america",
    "united kingdom": "europe",
    "france": "europe",
    "germany": "europe",
    "spain": "europe",
    "italy": "europe",
    "china": "asia",
    "japan": "asia",
    "india": "asia",
    "south korea": "asia",
    "australia": "oceania",
    "new zealand": "oceania",
    "egypt": "africa",
    "south africa": "africa",
    "nigeria": "africa",
    "kenya": "africa",
}


def get_continent_for_country(country: str) -> Optional[str]:
    """
    Get continent for a given country.
    
    Args:
        country: Country name
    
    Returns:
        Continent name or None
    """
    try:
        country_norm = normalize_country_name(country)
        return coco.convert(names=country, to='continent')
    except:      
        return COUNTRY_TO_CONTINENT.get(country_norm)


def format_distance(distance_km: float) -> str:
    """
    Format distance in human-readable form.
    
    Args:
        distance_km: Distance in kilometers
    
    Returns:
        Formatted string
    """
    if distance_km < 1:
        return f"{distance_km * 1000:.0f} m"
    elif distance_km < 100:
        return f"{distance_km:.1f} km"
    else:
        return f"{distance_km:.0f} km"

if __name__ == "__main__":
    # Test functions
    print("Testing haversine_distance:")
    # Madison, WI to Chicago, IL
    dist = haversine_distance(43.0731, -89.4012, 41.8781, -87.6298)
    print(f"  Madison to Chicago: {format_distance(dist)}")
    
    print("\nTesting parse_coordinates_from_text:")
    test_texts = [
        "Country: USA, Coordinates: 43.1278, -89.3898",
        "The location is at Latitude: 43.1278, Longitude: -89.3898",
        "Final answer: (43.1278, -89.3898)"
    ]
    for text in test_texts:
        coords = parse_coordinates_from_text(text)
        print(f"  '{text[:50]}...' -> {coords}")
    
    print("\nTesting countries_match:")
    test_pairs = [
        ("USA", "United States"),
        ("United Kingdom", "UK"),
        ("France", "France"),
    ]
    for p, g in test_pairs:
        match = countries_match(p, g)
        print(f"  '{p}' vs '{g}': {match}")