"""
Live Traffic Service — real-time road congestion from the TomTom Traffic API.

How it works:
  1. Every monitored location has GPS coordinates (lat/lng):
       - 24 ML junctions (INT_01..INT_24) from INTERSECTIONS
       - ~84 famous locations (ISB_01, KHI_01, ...) from LIVE_JUNCTIONS
  2. With a TomTom API key configured, the service queries the Traffic Flow
     Segment API for the road at that point and reads currentSpeed vs
     freeFlowSpeed → real congestion percentage right now.
  3. Without a key (or if the API fails), it falls back to a deterministic
     time-of-day estimator so the dashboard always works.

Getting a FREE key (2500 requests/day):
  https://developer.tomtom.com  →  Sign up  →  Create an app  →  copy key.

The key is read from (in order):
  - TOMTOM_API_KEY environment variable
  - live_api_key.json in the project root (saved from the dashboard)

Responses are cached for CACHE_TTL seconds per location so even browsing
every city stays far inside the free quota.
"""

import os
import json
import math
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.pk_config import (
    INTERSECTIONS, NODE_IDS, iter_live_junctions, live_junction_map,
)

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_FILE = os.path.join(PROJ_ROOT, "live_api_key.json")

CACHE_TTL = 300         # seconds per location (quota-safe: max ~288 req/day/location)
REQUEST_TIMEOUT = 8     # seconds per HTTP request

_DEFAULT_RUSH = {"rush_mult": 1.0, "night_mult": 0.15}


def _friendly_error(exc: Exception) -> str:
    """Turn a TomTom failure into an actionable message for the dashboard."""
    if isinstance(exc, urllib.error.HTTPError):
        code = exc.code
        if code == 401:
            return ("TomTom rejected the key (401 Unauthorized). Your account/key is "
                    "not activated yet. Verify your email at my.tomtom.com and make "
                    "sure the key has products assigned, then paste it again.")
        if code == 403:
            return ("Key not authorized for this service (403). Add the Traffic API "
                    "product to your key at my.tomtom.com.")
        if code == 429:
            return "TomTom daily/minute limit reached (429). Try again later."
        return f"TomTom returned HTTP {code}."
    if isinstance(exc, urllib.error.URLError):
        return f"Could not reach TomTom (network): {exc.reason}"
    return f"Unexpected error: {exc}"


def _level_from_ratio(ratio: float) -> str:
    """ratio = currentSpeed / freeFlowSpeed (0..1) -> congestion level."""
    if ratio >= 0.75:
        return "Low"
    if ratio >= 0.45:
        return "Medium"
    return "High"


class LiveTrafficService:
    """TomTom-backed live congestion with deterministic fallback."""

    def __init__(self):
        self._cache = {}   # loc_id -> (monotonic_ts, payload)
        self._locs = {}    # loc_id -> {name, city, lat, lng, rush_mult, night_mult}
        self._build_registry()
        self.api_key = self._load_key()
        self.key_valid = None   # None=untested, True=verified working, False=rejected
        self.last_error = ""
        self.estimator = None   # optional callable(loc) -> AI congestion dict

    def _build_registry(self):
        """ML intersections + famous live locations in one lookup table."""
        for nid in NODE_IDS:
            info = INTERSECTIONS[nid]
            self._locs[nid] = {
                "id": nid,
                "name": info["name"], "city": info["city"],
                "lat": info["lat"], "lng": info["lng"],
                "rush_mult": info.get("rush_mult", 1.0),
                "night_mult": info.get("night_mult", 0.15),
            }
        # City-level rush profile reused for its famous locations
        city_profile = {}
        for nid in NODE_IDS:
            info = INTERSECTIONS[nid]
            city_profile.setdefault(info["city"], {
                "rush_mult": info.get("rush_mult", 1.0),
                "night_mult": info.get("night_mult", 0.15),
            })
        for j in iter_live_junctions():
            prof = city_profile.get(j["city"], _DEFAULT_RUSH)
            self._locs[j["id"]] = {
                "id": j["id"],
                "name": j["name"], "city": j["city"],
                "lat": j["lat"], "lng": j["lng"],
                "rush_mult": prof["rush_mult"],
                "night_mult": prof["night_mult"],
            }

    # ── API key management ──────────────────────────────────────────────
    def _load_key(self) -> str:
        env = os.environ.get("TOMTOM_API_KEY", "").strip()
        if env:
            return env
        try:
            with open(KEY_FILE, "r") as f:
                return json.load(f).get("tomtom_api_key", "").strip()
        except Exception:
            return ""

    def set_estimator(self, fn):
        """Register an AI estimate callback (loc -> congestion dict) used when
        no live provider covers the location."""
        self.estimator = fn
        self._cache.clear()

    def set_key(self, key: str):
        self.api_key = (key or "").strip()
        try:
            with open(KEY_FILE, "w") as f:
                json.dump({"tomtom_api_key": self.api_key}, f)
        except Exception:
            pass
        self._cache.clear()
        self.key_valid = None
        self.last_error = ""

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    @property
    def mode(self) -> str:
        """'live' only when a key exists AND real data actually came through."""
        return "live" if (self.has_key and self.key_valid is True) else "estimated"

    # ── TomTom flow segment ─────────────────────────────────────────────
    def _fetch_tomtom(self, loc: dict) -> dict:
        point = f"{loc['lat']},{loc['lng']}"
        url = (
            "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?"
            + urllib.parse.urlencode({"key": self.api_key, "point": point})
        )
        req = urllib.request.Request(url, headers={"User-Agent": "ai-traffic-dashboard/1.0"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        fs = data.get("flowSegmentData")
        if not fs:
            raise ValueError("empty flowSegmentData")

        self.key_valid = True
        self.last_error = ""
        cur = float(fs.get("currentSpeed", 0))
        free = float(fs.get("freeFlowSpeed", 0)) or 1.0
        ratio = max(0.0, min(1.0, cur / free))
        return {
            "source": "tomtom-live",
            "live": True,
            "current_speed_kmh": round(cur, 1),
            "freeflow_speed_kmh": round(free, 1),
            "congestion_pct": round((1.0 - ratio) * 100, 1),
            "congestion_level": _level_from_ratio(ratio),
            "confidence": float(fs.get("confidence", 0.9)),
            "road_closed": bool(fs.get("roadClosure", False)),
        }

    # ── Deterministic fallback (no key / API failure) ───────────────────
    def _estimate(self, loc: dict) -> dict:
        # Prefer the AI model estimate when the backend provided one
        if self.estimator is not None:
            try:
                ai = self.estimator(loc)
                if ai:
                    ai["source"] = "ai-estimate"
                    ai["live"] = False
                    ai.setdefault("freeflow_speed_kmh", 55.0)
                    ai.setdefault("road_closed", False)
                    ai.setdefault("confidence", 0.75)
                    return ai
            except Exception as e:
                print(f"[LiveTraffic] estimator error for {loc.get('id')}: {e}")

        now = datetime.now()
        t = now.hour + now.minute / 60.0
        dow = now.weekday()

        morning = math.exp(-0.5 * ((t - 9.0) / 1.2) ** 2)
        evening = math.exp(-0.5 * ((t - 18.0) / 1.5) ** 2)
        lunch = 0.3 * math.exp(-0.5 * ((t - 13.0) / 1.0) ** 2)
        night = loc.get("night_mult", 0.15) if t <= 5 else 1.0
        pattern = max(morning * loc["rush_mult"], evening * loc["rush_mult"], lunch + 0.4) * night
        if dow == 6:
            pattern *= 0.55
        elif dow == 5:
            pattern *= 0.70

        load = min(1.0, pattern / loc["rush_mult"])
        speed = max(5.0, 55.0 - load * 50.0)
        return {
            "source": "estimated",
            "live": False,
            "current_speed_kmh": round(speed, 1),
            "freeflow_speed_kmh": 55.0,
            "congestion_pct": round(load * 100, 1),
            "congestion_level": _level_from_ratio(1.0 - load),
            "confidence": 0.6,
            "road_closed": False,
        }

    # ── Public API ──────────────────────────────────────────────────────
    def get_live(self, loc_id: str, force: bool = False) -> dict:
        loc = self._locs.get(loc_id)
        if loc is None:
            raise KeyError(f"Unknown location: {loc_id}")

        now_ts = time.time()
        cached = self._cache.get(loc_id)
        if not force and cached and now_ts - cached[0] < CACHE_TTL:
            return cached[1]

        payload = None
        if self.has_key:
            try:
                payload = self._fetch_tomtom(loc)
            except Exception as exc:
                self.key_valid = False
                self.last_error = _friendly_error(exc)
                payload = None
        if payload is None:
            payload = self._estimate(loc)

        payload["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload["junction"] = loc["name"]
        payload["city"] = loc["city"]
        self._cache[loc_id] = (now_ts, payload)
        return payload

    def _fetch_many(self, loc_ids: list) -> dict:
        out = {}
        with ThreadPoolExecutor(max_workers=min(8, len(loc_ids))) as ex:
            futures = {ex.submit(self.get_live, lid): lid for lid in loc_ids}
            for fut in as_completed(futures):
                lid = futures[fut]
                try:
                    out[lid] = fut.result()
                except Exception:
                    out[lid] = self._estimate(self._locs[lid])
        return out

    def get_all(self) -> dict:
        """Every ML junction (INT_*) — used by the legacy /api/v1/live panel."""
        return self._fetch_many(list(NODE_IDS))

    def get_cities(self) -> list:
        """City catalog for the live-location browser."""
        cities = []
        for city, juncs in live_junction_map_by_city():
            cities.append({"city": city, "count": len(juncs)})
        return cities

    def get_city(self, city: str) -> dict:
        """All famous locations in one city."""
        ids = [j["id"] for j in iter_live_junctions() if j["city"] == city]
        if not ids:
            raise KeyError(f"Unknown city: {city}")
        data = self._fetch_many(ids)
        return {
            "city": city,
            "mode": self.mode,
            "locations": [data[i] | {"junction_id": i} for i in ids],
        }

    def get_junction(self, junction_id: str) -> dict:
        """One famous location by id (e.g. ISB_01)."""
        return self.get_live(junction_id) | {"junction_id": junction_id}

    def test_key(self) -> dict:
        """Force-refresh one busy junction to verify the key actually works."""
        probe = self.get_live("ISB_01", force=True)
        working = bool(probe.get("live"))
        return {
            "working": working,
            "mode": self.mode,
            "error": "" if working else self.last_error,
            "probe": probe,
        }


def live_junction_map_by_city():
    """[(city, [junction dicts])] preserving pk_config declaration order."""
    from collections import OrderedDict
    grouped = OrderedDict()
    for j in iter_live_junctions():
        grouped.setdefault(j["city"], []).append(j)
    return list(grouped.items())
