#!/usr/bin/env python3
"""
Evaluation script for fine-tuned Qwen2-VL geo-reasoning model.
Computes ROUGE scores for reasoning text and haversine distance for coordinates.

Usage:
    python eval.py --base_model path/to/model --adapter path/to/adapter --test_data test.jsonl
"""

import argparse
import json
import re
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from predict import GeoPredictor
from rouge import Rouge


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
    
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in kilometers
    r = 6371
    
    return c * r


def parse_prediction(prediction_text: str) -> Dict:
    """
    Parse prediction text to extract country, city, continent, lat, lon.
    
    Expected format examples:
        "... Country: United States, Continent: North America, Coordinates: 43.1278, -89.3898"
        "... Country: France, City: Paris, Latitude: 48.8566, Longitude: 2.3522"
    
    Returns:
        dict with keys: country, city, continent, latitude, longitude, reasoning
    """
    result = {
        "country": None,
        "city": None,
        "continent": None,
        "latitude": None,
        "longitude": None,
        "reasoning": prediction_text  # Keep full text for ROUGE
    }
    
    # Extract country
    country_match = re.search(r'Country:\s*([^,\n]+)', prediction_text, re.IGNORECASE)
    if country_match:
        result["country"] = country_match.group(1).strip()
    
    # Extract city
    city_match = re.search(r'City:\s*([^,\n]+)', prediction_text, re.IGNORECASE)
    if city_match:
        result["city"] = city_match.group(1).strip()
    
    # Extract continent
    continent_match = re.search(r'Continent:\s*([^,\n]+)', prediction_text, re.IGNORECASE)
    if continent_match:
        result["continent"] = continent_match.group(1).strip()
    
    # Extract coordinates (various formats)
    # Format 1: "Coordinates: lat, lon"
    coord_match = re.search(r'Coordinates?:\s*([-\d.]+)\s*,\s*([-\d.]+)', prediction_text, re.IGNORECASE)
    if coord_match:
        result["latitude"] = float(coord_match.group(1))
        result["longitude"] = float(coord_match.group(2))
    else:
        # Format 2: "Latitude: X, Longitude: Y"
        lat_match = re.search(r'Latitude:\s*([-\d.]+)', prediction_text, re.IGNORECASE)
        lon_match = re.search(r'Longitude:\s*([-\d.]+)', prediction_text, re.IGNORECASE)
        if lat_match and lon_match:
            result["latitude"] = float(lat_match.group(1))
            result["longitude"] = float(lon_match.group(1))
    
    return result


def compute_country_accuracy(pred_country: Optional[str], 
                             gt_country: Optional[str],
                             threshold_km: float = 750) -> Dict:
    """
    Check if predicted country matches ground truth.
    
    Args:
        pred_country: Predicted country name
        gt_country: Ground truth country name
        threshold_km: Distance threshold for "close enough" (not used for exact match)
    
    Returns:
        dict with exact_match, normalized_match
    """
    if not pred_country or not gt_country:
        return {"exact_match": False, "normalized_match": False}
    
    # Exact match (case insensitive)
    exact = pred_country.lower().strip() == gt_country.lower().strip()
    
    # Normalized match (handle common variations)
    pred_norm = pred_country.lower().strip().replace("the ", "")
    gt_norm = gt_country.lower().strip().replace("the ", "")
    normalized = pred_norm == gt_norm or pred_norm in gt_norm or gt_norm in pred_norm
    
    return {
        "exact_match": exact,
        "normalized_match": normalized
    }


def compute_city_accuracy(pred_city: Optional[str],
                          gt_city: Optional[str],
                          pred_lat: Optional[float],
                          pred_lon: Optional[float],
                          gt_lat: float,
                          gt_lon: float,
                          threshold_km: float = 25) -> Dict:
    """
    Check if predicted city is correct or within distance threshold.
    
    Args:
        pred_city: Predicted city name
        gt_city: Ground truth city name (may be None)
        pred_lat, pred_lon: Predicted coordinates
        gt_lat, gt_lon: Ground truth coordinates
        threshold_km: Distance threshold for "close enough"
    
    Returns:
        dict with exact_match, within_threshold
    """
    result = {
        "exact_match": False,
        "within_threshold": False,
        "distance_km": None
    }
    
    # Check name match if both available
    if pred_city and gt_city:
        pred_norm = pred_city.lower().strip()
        gt_norm = gt_city.lower().strip()
        result["exact_match"] = pred_norm == gt_norm or pred_norm in gt_norm or gt_norm in pred_norm
    
    # Check distance if coordinates available
    if pred_lat is not None and pred_lon is not None:
        distance = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)
        result["distance_km"] = distance
        result["within_threshold"] = distance <= threshold_km
    
    return result


def compute_coordinate_accuracy(pred_lat: Optional[float],
                                pred_lon: Optional[float],
                                gt_lat: float,
                                gt_lon: float,
                                thresholds: List[int] = [1, 25, 200, 750, 2500]) -> Dict:
    """
    Compute distance error and accuracy at various thresholds.
    
    Args:
        pred_lat, pred_lon: Predicted coordinates
        gt_lat, gt_lon: Ground truth coordinates
        thresholds: List of distance thresholds in km
    
    Returns:
        dict with distance_km and accuracy at each threshold
    """
    if pred_lat is None or pred_lon is None:
        return {
            "distance_km": None,
            "within_1km": False,
            "within_25km": False,
            "within_200km": False,
            "within_750km": False,
            "within_2500km": False
        }
    
    distance = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)
    
    result = {"distance_km": distance}
    for threshold in thresholds:
        result[f"within_{threshold}km"] = distance <= threshold
    
    return result


def compute_rouge_scores(prediction: str, reference: str) -> Dict:
    """
    Compute ROUGE scores for reasoning text.
    
    Args:
        prediction: Predicted reasoning text
        reference: Ground truth reasoning text
    
    Returns:
        dict with ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    if not prediction or not reference:
        return {
            "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
            "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
            "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}
        }
    
    try:
        rouge = Rouge()
        scores = rouge.get_scores(prediction, reference)[0]
        return scores
    except Exception as e:
        print(f" ROUGE computation failed: {e}")
        return {
            "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
            "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
            "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}
        }


def evaluate_single_prediction(prediction_result: Dict, ground_truth: Dict) -> Dict:
    """
    Evaluate a single prediction against ground truth.
    
    Args:
        prediction_result: dict from predictor with 'prediction' key
        ground_truth: dict with country, city, latitude, longitude, reason
    
    Returns:
        dict with all evaluation metrics
    """
    # Parse prediction
    parsed = parse_prediction(prediction_result.get("prediction", ""))
    
    # Extract ground truth
    gt_country = ground_truth.get("country")
    gt_city = ground_truth.get("city")
    gt_lat = ground_truth.get("latitude")
    gt_lon = ground_truth.get("longitude")
    gt_reason = ground_truth.get("reason", "")
    
    # Compute metrics
    metrics = {
        "parsed": parsed,
        "country": compute_country_accuracy(parsed["country"], gt_country),
        "coordinate": compute_coordinate_accuracy(
            parsed["latitude"], parsed["longitude"], gt_lat, gt_lon
        ),
        "rouge": compute_rouge_scores(parsed["reasoning"], gt_reason)
    }
    
    # City evaluation if available
    if gt_city:
        metrics["city"] = compute_city_accuracy(
            parsed["city"], gt_city,
            parsed["latitude"], parsed["longitude"],
            gt_lat, gt_lon
        )
    
    return metrics


def aggregate_metrics(all_metrics: List[Dict]) -> Dict:
    """
    Aggregate metrics across all predictions.
    
    Args:
        all_metrics: List of metric dicts from evaluate_single_prediction
    
    Returns:
        dict with averaged/aggregated metrics
    """
    n = len(all_metrics)
    if n == 0:
        return {}
    
    aggregated = {
        "num_samples": n,
        "country": {
            "exact_match": 0.0,
            "normalized_match": 0.0
        },
        "coordinate": {
            "mean_distance_km": 0.0,
            "median_distance_km": 0.0,
            "within_1km": 0.0,
            "within_25km": 0.0,
            "within_200km": 0.0,
            "within_750km": 0.0,
            "within_2500km": 0.0
        },
        "rouge": {
            "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
            "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
            "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}
        }
    }
    
    # Country accuracy
    aggregated["country"]["exact_match"] = sum(
        m["country"]["exact_match"] for m in all_metrics
    ) / n * 100
    aggregated["country"]["normalized_match"] = sum(
        m["country"]["normalized_match"] for m in all_metrics
    ) / n * 100
    
    # Coordinate accuracy
    distances = [m["coordinate"]["distance_km"] for m in all_metrics 
                 if m["coordinate"]["distance_km"] is not None]
    if distances:
        aggregated["coordinate"]["mean_distance_km"] = sum(distances) / len(distances)
        aggregated["coordinate"]["median_distance_km"] = sorted(distances)[len(distances) // 2]
    
    for threshold in ["1km", "25km", "200km", "750km", "2500km"]:
        aggregated["coordinate"][f"within_{threshold}"] = sum(
            m["coordinate"][f"within_{threshold}"] for m in all_metrics
        ) / n * 100
    
    # ROUGE scores
    for rouge_type in ["rouge-1", "rouge-2", "rouge-l"]:
        for metric in ["f", "p", "r"]:
            aggregated["rouge"][rouge_type][metric] = sum(
                m["rouge"][rouge_type][metric] for m in all_metrics
            ) / n
    
    # City accuracy if available
    city_metrics = [m.get("city") for m in all_metrics if "city" in m]
    if city_metrics:
        aggregated["city"] = {
            "exact_match": sum(m["exact_match"] for m in city_metrics) / len(city_metrics) * 100,
            "within_25km": sum(m["within_threshold"] for m in city_metrics) / len(city_metrics) * 100
        }
    
    return aggregated


def print_results(aggregated: Dict):
    """Pretty print evaluation results."""
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\n🔢 Number of samples: {aggregated['num_samples']}")
    
    print("\n🌍 COUNTRY ACCURACY:")
    print(f"  Exact match:      {aggregated['country']['exact_match']:.2f}%")
    print(f"  Normalized match: {aggregated['country']['normalized_match']:.2f}%")
    
    if "city" in aggregated:
        print("\n🏙️  CITY ACCURACY:")
        print(f"  Exact match:  {aggregated['city']['exact_match']:.2f}%")
        print(f"  Within 25km:  {aggregated['city']['within_25km']:.2f}%")
    
    print("\n📍 COORDINATE ACCURACY:")
    print(f"  Mean distance:   {aggregated['coordinate']['mean_distance_km']:.2f} km")
    print(f"  Median distance: {aggregated['coordinate']['median_distance_km']:.2f} km")
    print(f"  Within 1 km:     {aggregated['coordinate']['within_1km']:.2f}%")
    print(f"  Within 25 km:    {aggregated['coordinate']['within_25km']:.2f}%")
    print(f"  Within 200 km:   {aggregated['coordinate']['within_200km']:.2f}%")
    print(f"  Within 750 km:   {aggregated['coordinate']['within_750km']:.2f}%")
    print(f"  Within 2500 km:  {aggregated['coordinate']['within_2500km']:.2f}%")
    
    print("\n📝 ROUGE SCORES (Reasoning Quality):")
    rouge = aggregated['rouge']
    print(f"  ROUGE-1: F1={rouge['rouge-1']['f']:.4f}, P={rouge['rouge-1']['p']:.4f}, R={rouge['rouge-1']['r']:.4f}")
    print(f"  ROUGE-2: F1={rouge['rouge-2']['f']:.4f}, P={rouge['rouge-2']['p']:.4f}, R={rouge['rouge-2']['r']:.4f}")
    print(f"  ROUGE-L: F1={rouge['rouge-l']['f']:.4f}, P={rouge['rouge-l']['p']:.4f}, R={rouge['rouge-l']['r']:.4f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2-VL geo-reasoning model")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to base Qwen2-VL model")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to fine-tuned adapter")
    
    # Data arguments
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to test JSONL file")
    parser.add_argument("--output", "-o", type=str, default="evaluation_results.json",
                        help="Path to save detailed results")
    
    # Evaluation parameters
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for testing)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧪 Qwen2-VL Geo-Reasoning Evaluation")
    print("=" * 80)
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter or 'None (base model only)'}")
    print(f"Test data: {args.test_data}")
    print("=" * 80)
    
    # Initialize predictor
    print("\n[1/3] Loading model...")
    predictor = GeoPredictor(args.base_model, args.adapter)
    
    # Run predictions and collect ground truth
    print("\n[2/3] Running predictions...")
    all_results = []
    all_metrics = []
    
    with open(args.test_data, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            if args.max_samples and idx > args.max_samples:
                break
            
            try:
                item = json.loads(line.strip())
                image_path = item.get("image_path")
                
                if not image_path:
                    continue
                
                print(f"  [{idx}] {image_path}")
                
                # Predict
                pred_result = predictor.predict(image_path)
                
                if "error" in pred_result:
                    print(f"    ✗ Error: {pred_result['error']}")
                    continue
                
                # Extract ground truth
                ground_truth = {
                    "country": item.get("country"),
                    "city": item.get("city"),
                    "continent": item.get("contient"),
                    "latitude": item.get("latitude"),
                    "longitude": item.get("longitude"),
                    "reason": item.get("reason", "")
                }
                
                # Evaluate
                metrics = evaluate_single_prediction(pred_result, ground_truth)
                
                # Store results
                all_results.append({
                    "image_path": image_path,
                    "prediction": pred_result["prediction"],
                    "ground_truth": ground_truth,
                    "metrics": metrics
                })
                all_metrics.append(metrics)
                
                # Print quick summary
                country_ok = "✓" if metrics["country"]["exact_match"] else "✗"
                dist = metrics["coordinate"]["distance_km"]
                dist_str = f"{dist:.1f} km" if dist is not None else "N/A"
                print(f"    Country: {country_ok}  Distance: {dist_str}")
                
            except Exception as e:
                print(f"  [{idx}] Error: {e}")
    
    # Aggregate metrics
    print(f"\n[3/3] Computing aggregate metrics...")
    aggregated = aggregate_metrics(all_metrics)
    
    # Print results
    print_results(aggregated)
    
    # Save detailed results
    output_data = {
        "summary": aggregated,
        "detailed_results": all_results
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {args.output}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()