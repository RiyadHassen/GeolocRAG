#!/usr/bin/env python3
"""
Parse image filenames to extract location data and create JSONL dataset.

Filename format: city_lat_long.jpg
Examples:
    - Madison_43.0731_-89.4012.jpg
    - New_York_40.7128_-74.0060.jpg
    - San_Francisco_37.7749_-122.4194.png

Usage:
    python parse_dataset.py --image_dir /path/to/images --output dataset.jsonl
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, List
from geopy.geocoders import Nominatim
import country_converter as coco


def parse_filename(filename: str) -> Optional[Dict]:
    """
    Parse filename to extract city, latitude, and longitude.
    
    Supported formats:
        - city_lat_long.ext              -> Madison_43.0731_-89.4012.jpg
        - city_(lat, long).ext            -> Albuquerque_(35.103183, -106.65144).jpg
        - city (lat, long).ext            -> New York (40.7128, -74.0060).jpg
        - city_lat_long_extra.ext         -> Paris_48.8566_2.3522_photo.jpg
    
    Args:
        filename: Image filename
    
    Returns:
        dict with city, latitude, longitude or None if parse fails
    """
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Pattern 1: city_(lat, lon) or city (lat, lon)
    # Matches: Albuquerque_(35.103183, -106.65144) or New York (40.7128, -74.0060)
    pattern1 = r'^(.+?)\s*[_]?\s*\(([-\d.]+)\s*,\s*([-\d.]+)\).*'
    match = re.match(pattern1, name_without_ext)
    
    if match:
        city = match.group(1).replace('_', ' ')  # Handle multi-word cities
        try:
            lat = float(match.group(2))
            lon = float(match.group(3))
            
            # Validate coordinates
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return {
                    "city": city,
                    "latitude": lat,
                    "longitude": lon
                }
        except ValueError:
            pass
    
    return None

def get_location_info(lat: float, lon: float, geolocator: Nominatim, cc: coco.CountryConverter) -> Dict:
    """
    Get country and continent from coordinates using reverse geocoding.
    
    Args:
        lat, lon: Coordinates
        geolocator: Nominatim geolocator instance
        cc: CountryConverter instance
    
    Returns:
        dict with country, country_code, continent
    """
    result = {
        "country": None,
        "country_code": None,
        "continent": None
    }
    
    try:
        # Reverse geocode
        location = geolocator.reverse(f"{lat}, {lon}", language="en", timeout=10)
        
        if location and location.raw.get("address"):
            address = location.raw["address"]
            
            # Get country
            result["country"] = address.get("country")
            result["country_code"] = address.get("country_code", "").upper()
            
            # Get continent from country code
            if result["country_code"]:
                try:
                    result["continent"] = cc.convert(result["country_code"], to="continent")
                except:
                    pass
    
    except Exception as e:
        print(f" Geocoding error for ({lat}, {lon}): {e}")
    
    return result


def process_images(image_dir: str, 
                   output_path: str, 
                   use_geocoding: bool = True,
                   max_samples: Optional[int] = None):
    """
    Process all images in directory and create JSONL dataset.
    
    Args:
        image_dir: Directory containing images
        output_path: Output JSONL file path
        use_geocoding: Whether to use reverse geocoding for country/continent
        max_samples: Maximum number of samples to process (for testing)
    """
    print("=" * 80)
    print("📸 Image Dataset Parser")
    print("=" * 80)
    print(f"Image directory: {image_dir}")
    print(f"Output file: {output_path}")
    print(f"Use geocoding: {use_geocoding}")
    print("=" * 80)
    
    # Initialize geocoder if needed
    geolocator = None
    cc = None
    if use_geocoding:
        print("\n🌍 Initializing geocoder...")
        geolocator = Nominatim(user_agent="qwen2vl_geo_parser")
        cc = coco.CountryConverter()
        print("✓ Geocoder ready")
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    image_dir_path = Path(image_dir)
    image_files = [
        f for f in image_dir_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    print(f"\n📁 Found {len(image_files)} image files")
    
    if max_samples:
        image_files = image_files[:max_samples]
        print(f"   Processing first {max_samples} files only")
    
    # Process each image
    processed = 0
    skipped = 0
    dataset = []
    
    print("\n🔄 Processing files...\n")
    
    for idx, image_path in enumerate(image_files, 1):
        filename = image_path.name
        print(f"[{idx}/{len(image_files)}] {filename}")
        
        # Parse filename
        parsed = parse_filename(filename)
        
        if not parsed:
            print(f"  ✗ Failed to parse filename")
            skipped += 1
            continue
        
        # Create data entry
        entry = {
            "image_path": str(image_path.absolute()),
            "city": parsed["city"],
            "latitude": parsed["latitude"],
            "longitude": parsed["longitude"]
        }
        
        # Add geocoding info if enabled
        if use_geocoding:
            print(f"  🌍 Geocoding {parsed['latitude']}, {parsed['longitude']}...")
            location_info = get_location_info(
                parsed["latitude"], 
                parsed["longitude"], 
                geolocator,
                cc
            )
            entry.update(location_info)
            
            if location_info["country"]:
                print(f"  ✓ {parsed['city']}, {location_info['country']} ({location_info['continent']})")
            else:
                print(f"  Could not determine country")
        else:
            print(f"  ✓ {parsed['city']} at {parsed['latitude']}, {parsed['longitude']}")
        
        dataset.append(entry)
        processed += 1
        print()
    
    # Save to JSONL
    print("=" * 80)
    print(f"💾 Saving dataset...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {processed} entries to {output_path}")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print("=" * 80)
    
    # Print sample
    if dataset:
        print("\n📋 Sample entry:")
        print(json.dumps(dataset[0], indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="Parse image filenames to create geo-reasoning dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--output", "-o", type=str, default="dataset.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--no-geocoding", action="store_true",
                        help="Skip reverse geocoding (faster, no country/continent)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    # Validate image directory
    if not os.path.isdir(args.image_dir):
        print(f"❌ Error: Image directory does not exist: {args.image_dir}")
        return
    
    # Process images
    process_images(
        image_dir=args.image_dir,
        output_path=args.output,
        use_geocoding=not args.no_geocoding,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()