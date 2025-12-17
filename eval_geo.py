#!/usr/bin/env python3
"""
Evaluation script for full GeoLoc RAG Pipeline.
Tests the complete pipeline: Reasoner -> RAG -> Final Predictor

Computes:
  1. Haversine distance (km)
  2. GeoGuessr score (0-5000 points)
  3. City name match (exact + geocoded validation)
  4. Country match

Usage:
    python eval_rag_pipeline.py \
        --rag_index_path /path/to/index \
        --rag_base /path/to/images \
        --reasoner_model_path /path/to/reasoner \
        --inference_model_path /path/to/inference \
        --inference_adapter_path /path/to/adapter \
        --test_data test.jsonl
"""

import argparse
import json
import math
import re
import gc
import torch
from typing import Dict, Optional, List
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import country_converter as coco
from sentence_transformers import SentenceTransformer

from GeoLocPipeline import GeoLocReasonPipeline


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance in kilometers."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * R


def calculate_geoguessr_score(distance_km: float) -> int:
    """
    Calculate GeoGuessr-style score based on distance.
    
    Args:
        distance_km: Distance in kilometers
    
    Returns:
        Score from 0 to 5000
    """
    if distance_km < 0:
        return 5000
    
    # Exponential decay formula similar to GeoGuessr
    scale_factor = 2000  # Adjust for desired curve
    score = int(5000 * math.exp(-distance_km / scale_factor))
    
    return max(0, min(5000, score))


def normalize_city_name(city: str) -> str:
    """Normalize city name for comparison."""
    if not city:
        return ""
    normalized = city.lower().strip()
    normalized = re.sub(r'\s+(city|town|village)$', '', normalized)
    return normalized


def cities_match(pred_city: str, gt_city: str, fuzzy: bool = True) -> bool:
    """Check if two city names match."""
    if not pred_city or not gt_city:
        return False
    
    pred_norm = normalize_city_name(pred_city)
    gt_norm = normalize_city_name(gt_city)
    
    # Exact match
    if pred_norm == gt_norm:
        return True
    
    # Fuzzy match
    if fuzzy:
        # Check substring
        if pred_norm in gt_norm or gt_norm in pred_norm:
            return True
        
        # Check if both contain the same words
        pred_words = set(pred_norm.split())
        gt_words = set(gt_norm.split())
        if pred_words & gt_words:  # Any common words
            return True
    
    return False


def get_location_from_coords(lat: float, lon: float, geolocator: Nominatim, 
                             cc: coco.CountryConverter) -> Dict:
    """Reverse geocode coordinates to get city, country, continent."""
    result = {
        "city": None,
        "country": None,
        "continent": None
    }
    
    try:
        location = geolocator.reverse(f"{lat}, {lon}", language="en", timeout=10)
        
        if location and location.raw.get("address"):
            address = location.raw["address"]
            
            # Get city
            for key in ["city", "town", "village", "municipality", "county"]:
                if key in address:
                    result["city"] = address[key]
                    break
            
            # Get country
            result["country"] = address.get("country")
            country_code = address.get("country_code", "").upper()
            
            # Get continent
            if country_code:
                try:
                    result["continent"] = cc.convert(country_code, to="continent")
                except:
                    pass
    
    except Exception as e:
        print(f" ⚠️  Geocoding error: {e}")
    
    return result


def evaluate_single_prediction(prediction: Dict,
                               gt_lat: float,
                               gt_lon: float,
                               gt_city: Optional[str],
                               gt_country: Optional[str],
                               geolocator: Nominatim,
                               cc: coco.CountryConverter) -> Dict:
    """Evaluate a single prediction."""
    
    if prediction is None or 'latitude' not in prediction or 'longitude' not in prediction:
        return {
            "distance_km": None,
            "geoguessr_score": 0,
            "city_match": False,
            "country_match": False,
            "pred_coords": None,
            "pred_location": None,
            "gt_location": None,
            "error": "Could not parse coordinates from prediction"
        }
    
    pred_lat, pred_lon = prediction['latitude'], prediction['longitude']
    
    # Calculate distance
    distance_km = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)
    
    # Calculate GeoGuessr score
    geoguessr_score = calculate_geoguessr_score(distance_km)
    
    # Parse predicted city and country
    pred_city = prediction.get('city')
    pred_country = prediction.get('country')
    
    # Get location info from ground truth coordinates if not provided
    gt_location = {
        "city": gt_city,
        "country": gt_country
    }
    
    # If gt_city or gt_country not provided, geocode the ground truth coords
    if not gt_city or not gt_country:
        gt_geocoded = get_location_from_coords(gt_lat, gt_lon, geolocator, cc)
        if not gt_city:
            gt_location["city"] = gt_geocoded["city"]
        if not gt_country:
            gt_location["country"] = gt_geocoded["country"]
    
    # Check city match
    city_match = False
    if pred_city and gt_location["city"]:
        city_match = cities_match(pred_city, gt_location["city"])
    
    # Also check if predicted coords are in the right city
    geocoded_city_match = False
    if pred_city and gt_city:
        geocoded_city_match = cities_match(pred_city, gt_city)
    
    # Check country match
    country_match = False
    if pred_country and gt_location["country"]:
        pred_norm = pred_country.lower().strip()
        gt_norm = gt_location["country"].lower().strip()
        country_match = pred_norm == gt_norm or pred_norm in gt_norm or gt_norm in pred_norm
    
    return {
        "distance_km": distance_km,
        "geoguessr_score": geoguessr_score,
        "city_match": city_match or geocoded_city_match,
        "city_name_match": city_match,
        "city_geocoded_match": geocoded_city_match,
        "country_match": country_match,
        "pred_coords": {"latitude": pred_lat, "longitude": pred_lon},
        "pred_city": pred_city,
        "pred_country": pred_country,
        "gt_location": gt_location
    }


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text


def parse_prediction(prediction_text: str) -> Dict:
    """
    Parse prediction text to extract country, city, continent, lat, lon.
    Supports JSON and free-text formats.
    """
    result = {
        "country": None,
        "city": None,
        "continent": None,
        "latitude": None,
        "longitude": None,
        "reasoning": prediction_text
    }

    # Try JSON parsing first
    try:
        prediction_text = _strip_code_fences(prediction_text)
        parsed = json.loads(prediction_text)

        # Handle nested {"result": {...}} or flat JSON
        data = parsed.get("result", parsed)

        if isinstance(data, dict):
            result["country"] = data.get("country")
            result["city"] = data.get("city")
            result["continent"] = data.get("continent")
            result["latitude"] = data.get("latitude")
            result["longitude"] = data.get("longitude")

            return result
    except (json.JSONDecodeError, TypeError):
        pass  # fall back to regex parsing

    # Regex fallback
    country_match = re.search(r'["\']?country["\']?\s*:\s*["\']([^"\']+)["\']', prediction_text, re.IGNORECASE)
    if country_match:
        result["country"] = country_match.group(1).strip()

    city_match = re.search(r'["\']?city["\']?\s*:\s*["\']([^"\']+)["\']', prediction_text, re.IGNORECASE)
    if city_match:
        result["city"] = city_match.group(1).strip()

    continent_match = re.search(r'["\']?continent["\']?\s*:\s*["\']([^"\']+)["\']', prediction_text, re.IGNORECASE)
    if continent_match:
        result["continent"] = continent_match.group(1).strip()

    lat_match = re.search(r'["\']?latitude["\']?\s*:\s*([-\d.]+)', prediction_text, re.IGNORECASE)
    lon_match = re.search(r'["\']?longitude["\']?\s*:\s*([-\d.]+)', prediction_text, re.IGNORECASE)
    
    if lat_match and lon_match:
        result["latitude"] = float(lat_match.group(1))
        result["longitude"] = float(lon_match.group(1))

    return result


def aggregate_metrics(all_results: List[Dict]) -> Dict:
    """Aggregate evaluation metrics."""
    n = len(all_results)
    if n == 0:
        return {}
    
    # Filter out errors
    valid_results = [r for r in all_results if r['metrics'].get("distance_km") is not None]
    n_valid = len(valid_results)
    
    if n_valid == 0:
        return {"error": "No valid predictions"}
    
    # Compute statistics
    distances = [r['metrics']["distance_km"] for r in valid_results]
    scores = [r['metrics']["geoguessr_score"] for r in valid_results]
    
    aggregated = {
        "total_samples": n,
        "valid_predictions": n_valid,
        "failed_predictions": n - n_valid,
        
        "distance_km": {
            "mean": sum(distances) / n_valid,
            "median": sorted(distances)[n_valid // 2],
            "min": min(distances),
            "max": max(distances),
            "std": (sum((d - sum(distances)/n_valid)**2 for d in distances) / n_valid) ** 0.5
        },
        
        "geoguessr_score": {
            "mean": sum(scores) / n_valid,
            "median": sorted(scores)[n_valid // 2],
            "total": sum(scores),
            "min": min(scores),
            "max": max(scores)
        },
        
        "city_accuracy": {
            "name_match": sum(r['metrics']["city_name_match"] for r in valid_results) / n_valid * 100,
            "geocoded_match": sum(r['metrics']["city_geocoded_match"] for r in valid_results) / n_valid * 100
        },
        
        "country_accuracy": {
            "match": sum(r['metrics']["country_match"] for r in valid_results) / n_valid * 100
        },
        
        "distance_thresholds": {
            "within_1km": sum(d <= 1 for d in distances) / n_valid * 100,
            "within_25km": sum(d <= 25 for d in distances) / n_valid * 100,
            "within_200km": sum(d <= 200 for d in distances) / n_valid * 100,
            "within_750km": sum(d <= 750 for d in distances) / n_valid * 100,
            "within_2500km": sum(d <= 2500 for d in distances) / n_valid * 100
        }
    }
    
    return aggregated


def print_results(aggregated: Dict):
    """Pretty print evaluation results."""
    print("\n" + "=" * 80)
    print("📊 RAG PIPELINE EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\n📈 Statistics:")
    print(f"  Total samples:       {aggregated['total_samples']}")
    print(f"  Valid predictions:   {aggregated['valid_predictions']}")
    print(f"  Failed predictions:  {aggregated['failed_predictions']}")
    
    dist = aggregated['distance_km']
    print(f"\n📏 Distance Errors:")
    print(f"  Mean:    {dist['mean']:.2f} km")
    print(f"  Median:  {dist['median']:.2f} km")
    print(f"  Std Dev: {dist['std']:.2f} km")
    print(f"  Range:   [{dist['min']:.2f}, {dist['max']:.2f}] km")
    
    score = aggregated['geoguessr_score']
    print(f"\n🎯 GeoGuessr Scores:")
    print(f"  Mean score:   {score['mean']:.0f} / 5000")
    print(f"  Median score: {score['median']:.0f} / 5000")
    print(f"  Total score:  {score['total']:.0f} / {aggregated['valid_predictions'] * 5000}")
    print(f"  Percentage:   {score['mean']/50:.1f}%")
    
    city = aggregated['city_accuracy']
    print(f"\n🏙️  City Accuracy:")
    print(f"  Name match:     {city['name_match']:.2f}%")
    print(f"  Geocoded match: {city['geocoded_match']:.2f}%")
    
    country = aggregated['country_accuracy']
    print(f"\n🌍 Country Accuracy:")
    print(f"  Match: {country['match']:.2f}%")
    
    thresh = aggregated['distance_thresholds']
    print(f"\n📍 Distance Thresholds:")
    print(f"  Within 1 km:     {thresh['within_1km']:.2f}%")
    print(f"  Within 25 km:    {thresh['within_25km']:.2f}%")
    print(f"  Within 200 km:   {thresh['within_200km']:.2f}%")
    print(f"  Within 750 km:   {thresh['within_750km']:.2f}%")
    print(f"  Within 2500 km:  {thresh['within_2500km']:.2f}%")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluation for GeoLoc RAG Pipeline")
    
    # RAG parameters
    parser.add_argument("--rag_index_path", type=str, required=True,
                       help="Path to RAG index file")
    parser.add_argument("--rag_base", type=str, required=True,
                       help="Base path for RAG images")
    
    # Model parameters
    parser.add_argument("--reasoner_model_path", type=str, 
                       default="./Qwen-VL/Qwen-VL-Models/Qwen-VL-Chat",
                       help="Path to reasoner model")
    parser.add_argument("--inference_model_path", type=str,
                       default="./Qwen-VL/Qwen-VL-Models/Qwen2-VL-Chat-Finetuned",
                       help="Path to inference base model")
    parser.add_argument("--inference_adapter_path", type=str,
                       default="./Qwen-VL/Qwen-VL-Adapters/qwen2-vl-chat-finetuned-geolocrag-adapter",
                       help="Path to inference adapter")
    
    # Test parameters
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to test JSONL file")
    parser.add_argument("--output", "-o", type=str, default="eval_rag_results.json",
                       help="Output JSON file")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to evaluate")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧪 GeoLoc RAG Pipeline Evaluation")
    print("=" * 80)
    print(f"RAG Index: {args.rag_index_path}")
    print(f"RAG Base: {args.rag_base}")
    print(f"Reasoner: {args.reasoner_model_path}")
    print(f"Inference: {args.inference_model_path}")
    print(f"Adapter: {args.inference_adapter_path}")
    print(f"Test data: {args.test_data}")
    print("=" * 80)
    
    # Initialize geocoder
    print("\n[1/4] Initializing geocoder...")
    geolocator = Nominatim(user_agent="qwen2vl_rag_eval")
    cc = coco.CountryConverter()
    print("✓ Geocoder ready")
    
    # Initialize RAG model
    print("\n[2/4] Loading RAG model...")
    rag_model = SentenceTransformer('clip-ViT-B-32', device='cpu')
    print("✓ RAG model ready")
    
    # Initialize pipeline
    print("\n[3/4] Initializing GeoLoc Pipeline...")
    pipeline = GeoLocReasonPipeline(
        rag_index_path=args.rag_index_path,
        rag_images_path=args.rag_base,
        rag_model=rag_model,
        reasoner_model_path=args.reasoner_model_path,
        inference_model_path=args.inference_model_path,
        inference_adapter_path=args.inference_adapter_path
    )
    print("✓ Pipeline ready")
    
    # Run evaluation
    print("\n[4/4] Running predictions...\n")
    all_results = []
    
    with open(args.test_data, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            if args.max_samples and idx > args.max_samples:
                break
            
            try:
                item = json.loads(line.strip())
                image_path = item.get("image_path")
                
                if not image_path:
                    continue
                
                print(f"\n{'='*80}")
                print(f"[{idx}] Processing: {image_path}")
                print('='*80)
                
                # Run full pipeline
                pred_text = pipeline.final_predication_based_on_result(image_path)
                
                # Parse prediction
                pred_result = parse_prediction(pred_text)
                
                if not pred_result.get('latitude') or not pred_result.get('longitude'):
                    print(f"    ✗ Failed to parse prediction")
                    all_results.append({
                        "image_path": image_path,
                        "prediction": pred_result,
                        "ground_truth": {
                            "latitude": item["latitude"],
                            "longitude": item["longitude"],
                            "city": item.get("city"),
                            "country": item.get("country")
                        },
                        "metrics": {
                            "distance_km": None,
                            "error": "Failed to parse coordinates"
                        }
                    })
                    continue
                
                # Evaluate
                metrics = evaluate_single_prediction(
                    prediction=pred_result,
                    gt_lat=item["latitude"],
                    gt_lon=item["longitude"],
                    gt_city=item.get("city"),
                    gt_country=item.get("country"),
                    geolocator=geolocator,
                    cc=cc
                )
                
                # Store result
                result = {
                    "image_path": image_path,
                    "prediction": pred_result,
                    "ground_truth": {
                        "latitude": item["latitude"],
                        "longitude": item["longitude"],
                        "city": item.get("city"),
                        "country": item.get("country")
                    },
                    "metrics": metrics
                }
                all_results.append(result)
                
                # Print quick summary
                if metrics.get("distance_km") is not None:
                    city_icon = "✓" if metrics["city_match"] else "✗"
                    country_icon = "✓" if metrics["country_match"] else "✗"
                    print(f"\n📊 Results:")
                    print(f"    Distance: {metrics['distance_km']:.1f} km")
                    print(f"    GeoGuessr Score: {metrics['geoguessr_score']}/5000")
                    print(f"    City Match: {city_icon} {pred_result.get('city')} vs {item.get('city')}")
                    print(f"    Country Match: {country_icon} {pred_result.get('country')} vs {item.get('country')}")
                else:
                    print(f"    ✗ {metrics.get('error', 'Unknown error')}")
                
                # Clean up memory after each prediction
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\n[{idx}] Error: {e}")
                import traceback
                traceback.print_exc()
    
    # Aggregate metrics
    print("\n" + "="*80)
    print("Computing aggregate metrics...")
    print("="*80)
    aggregated = aggregate_metrics(all_results)
    
    # Print results
    print_results(aggregated)
    
    # Save results
    output_data = {
        "summary": aggregated,
        "detailed_results": all_results
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {args.output}")
    print("\n Evaluation complete!")


if __name__ == "__main__":
    main()