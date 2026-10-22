"""
Sample Data Generator — Generates realistic Pakistani traffic data.
24 junctions × 4 lanes × 56 days × 5-minute intervals.
"""

import csv
import os
import sys
import random
import math
from datetime import datetime, timedelta

# Allow importing the shared Pakistan config from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pk_config import INTERSECTIONS, LANE_SHARES, LANES, DEFAULT_CSV

LANE_DISTRIBUTION = {"Lane_N": 0.30, "Lane_S": 0.25, "Lane_E": 0.25, "Lane_W": 0.20}


def time_multiplier(hour, minute, is_weekend):
    t = hour + minute / 60.0
    if is_weekend:
        if 10 <= t < 13:
            return random.uniform(0.5, 0.7)
        elif 17 <= t < 20:
            return random.uniform(0.45, 0.65)
        elif 0 <= t < 6:
            return random.uniform(0.05, 0.15)
        return random.uniform(0.25, 0.45)
    else:
        if 8 <= t < 10:
            return random.uniform(0.8, 1.0)
        elif 10 <= t < 12:
            return random.uniform(0.45, 0.65)
        elif 12 <= t < 14:
            return random.uniform(0.5, 0.7)
        elif 14 <= t < 17:
            return random.uniform(0.45, 0.6)
        elif 17 <= t < 20:
            return random.uniform(0.85, 1.0)
        elif 20 <= t < 22:
            return random.uniform(0.3, 0.5)
        elif 0 <= t < 6:
            return random.uniform(0.05, 0.15)
        return random.uniform(0.25, 0.45)


def classify_congestion(vehicle_count, avg_speed):
    if avg_speed < 10 or vehicle_count > 45:
        return "High"
    elif avg_speed < 18 or vehicle_count > 25:
        return "Medium"
    return "Low"


def generate_dataset(output_path, num_days=56):
    rows = []
    start_date = datetime(2025, 1, 1)

    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        is_weekend = current_date.weekday() >= 5

        for hour in range(24):
            for minute in range(0, 60, 5):
                ts = current_date.replace(hour=hour, minute=minute, second=0)

                for int_id, info in INTERSECTIONS.items():
                    mult = time_multiplier(hour, minute, is_weekend)
                    total_vehicles = int(info["base_count"] * mult + random.randint(-10, 10))
                    total_vehicles = max(0, total_vehicles)

                    shares = LANE_SHARES.get(int_id, LANE_DISTRIBUTION)
                    for lane in LANES:
                        share = shares.get(lane, 0.25)
                        lane_noise = random.uniform(-0.08, 0.08)
                        lane_count = int(total_vehicles * (share + lane_noise))
                        lane_count = max(0, lane_count)

                        speed_factor = max(0.25, 1.0 - (lane_count / (info["base_count"] * 0.4)))
                        avg_speed = round(info["base_speed"] * speed_factor + random.uniform(-3, 3), 1)
                        avg_speed = max(2.0, avg_speed)

                        congestion = classify_congestion(lane_count, avg_speed)

                        rows.append({
                            "intersection_id": int_id,
                            "intersection_name": info["name"],
                            "city": info["city"],
                            "lane_id": lane,
                            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                            "vehicle_count": lane_count,
                            "avg_speed": avg_speed,
                            "congestion_level": congestion,
                        })

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "intersection_id", "intersection_name", "city", "lane_id",
            "timestamp", "vehicle_count", "avg_speed", "congestion_level"
        ])
        writer.writeheader()
        writer.writerows(rows)

    cities = {i["city"] for i in INTERSECTIONS.values()}
    print(f"Generated {len(rows):,} records ({num_days} days, {len(INTERSECTIONS)} junctions, {len(LANES)} lanes)")
    print(f"Cities: {', '.join(sorted(cities))}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generate_dataset(os.path.join(script_dir, DEFAULT_CSV))