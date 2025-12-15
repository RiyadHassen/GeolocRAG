import requests, json, os
from typing import Dict

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
CACHE_PATH = "RAG/overpass_cache.json" # local cache file

def _load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_cache(cache: Dict[str, str]):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def overpass_tags_nearby(lat: float, lon: float, radius_m: int = 250):
    # round coords for caching; nearby points share similar context
    key = f"{round(lat,4)},{round(lon,4)},{radius_m}"
    cache = _load_cache()
    if key in cache:
        return cache[key]

    query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius_m},{lat},{lon});
      way(around:{radius_m},{lat},{lon});
    );
    out tags 200;
    """

    try:
        r = requests.post(OVERPASS_URL, data=query.encode("utf-8"), timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return f"OSM_ERROR: {e}"

    allowed = (
        "name","amenity","shop","tourism","historic","building",
        "highway","surface","natural","landuse","place","religion"
    )

    lines = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        keep = {k: tags[k] for k in allowed if k in tags}
        if keep:
            lines.append(str(keep))

    text = "OSM: no informative tags found nearby." if not lines else \
           "OSM nearby tags:\n" + "\n".join(lines[:60])

    cache[key] = text
    _save_cache(cache)
    return text

def build_geo_context(lat: float, lon: float):
    return overpass_tags_nearby(lat, lon, radius_m=250)
