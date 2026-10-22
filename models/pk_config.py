"""
Pakistan Road-Network Configuration.

Single source of truth for all intersections/junctions used across the
whole system (prediction, forecasting, detection and signal control).

  - INTERSECTIONS : junction metadata + deterministic traffic characteristics
  - LANE_SHARES   : per-intersection lane distribution (must sum to 1)
  - NODE_IDS      : ordered junction ids (order defines the corridor backbone
                    used by the graph forecaster for spatial fusion)
"""

LANES = ["Lane_N", "Lane_S", "Lane_E", "Lane_W"]

# name / city / characteristics for the deterministic time-of-day estimator
# base_traffic: baseline volume | rush_mult: morning/evening rush boost
# night_mult: night valley factor | base_count/base_speed: data generation
# lat/lng: junction GPS coordinates used by the live traffic provider
INTERSECTIONS = {
    "INT_01": {"name": "Zero Point Interchange", "city": "Islamabad",   "base_traffic": 42, "rush_mult": 2.2, "night_mult": 0.15, "base_count": 135, "base_speed": 24, "lat": 33.6844, "lng": 73.0479},
    "INT_02": {"name": "Raja Bazaar Chowk",      "city": "Rawalpindi",  "base_traffic": 46, "rush_mult": 2.2, "night_mult": 0.17, "base_count": 150, "base_speed": 22, "lat": 33.5658, "lng": 73.0707},
    "INT_03": {"name": "Chowk Yadgar",           "city": "Peshawar",    "base_traffic": 44, "rush_mult": 2.3, "night_mult": 0.14, "base_count": 160, "base_speed": 20, "lat": 34.0078, "lng": 71.5838},
    "INT_04": {"name": "Kot Chowk",              "city": "Mardan",      "base_traffic": 32, "rush_mult": 2.1, "night_mult": 0.16, "base_count": 120, "base_speed": 24, "lat": 34.1980, "lng": 72.0230},
    "INT_05": {"name": "Main Bazaar Chowk",      "city": "Swabi",       "base_traffic": 26, "rush_mult": 2.0, "night_mult": 0.18, "base_count": 105, "base_speed": 26, "lat": 34.1203, "lng": 72.4697},
    "INT_06": {"name": "Mansehra Chowk",         "city": "Abbottabad",  "base_traffic": 28, "rush_mult": 2.0, "night_mult": 0.18, "base_count": 110, "base_speed": 26, "lat": 34.1688, "lng": 73.2215},
    "INT_07": {"name": "Sarband Chowk",          "city": "Charsadda",   "base_traffic": 25, "rush_mult": 2.0, "night_mult": 0.18, "base_count": 100, "base_speed": 26, "lat": 34.1450, "lng": 71.7310},
    "INT_08": {"name": "Rashakai Chowk",         "city": "Nowshera",    "base_traffic": 27, "rush_mult": 2.0, "night_mult": 0.17, "base_count": 105, "base_speed": 25, "lat": 34.0150, "lng": 71.9740},
    "INT_09": {"name": "Stadium Chowk",          "city": "Kohat",       "base_traffic": 26, "rush_mult": 2.0, "night_mult": 0.18, "base_count": 100, "base_speed": 25, "lat": 33.5860, "lng": 71.4420},
    "INT_10": {"name": "Chauburji Chowk",        "city": "Lahore",      "base_traffic": 52, "rush_mult": 2.5, "night_mult": 0.12, "base_count": 185, "base_speed": 18, "lat": 31.5685, "lng": 74.3083},
    "INT_11": {"name": "Clock Tower Chowk",      "city": "Faisalabad",  "base_traffic": 46, "rush_mult": 2.3, "night_mult": 0.13, "base_count": 170, "base_speed": 20, "lat": 31.4185, "lng": 73.0789},
    "INT_12": {"name": "G.T. Road Chowk",        "city": "Gujranwala",  "base_traffic": 40, "rush_mult": 2.2, "night_mult": 0.15, "base_count": 150, "base_speed": 22, "lat": 32.1610, "lng": 74.1880},
    "INT_13": {"name": "Airlines Chowk",         "city": "Sialkot",     "base_traffic": 38, "rush_mult": 2.1, "night_mult": 0.16, "base_count": 140, "base_speed": 22, "lat": 32.4945, "lng": 74.5229},
    "INT_14": {"name": "G.T. Road Chowk",        "city": "Jhelum",      "base_traffic": 30, "rush_mult": 2.0, "night_mult": 0.17, "base_count": 115, "base_speed": 24, "lat": 32.9424, "lng": 73.7257},
    "INT_15": {"name": "Kutchery Chowk",         "city": "Sargodha",    "base_traffic": 33, "rush_mult": 2.1, "night_mult": 0.16, "base_count": 120, "base_speed": 23, "lat": 32.0836, "lng": 72.6711},
    "INT_16": {"name": "Kumharanwala Chowk",     "city": "Multan",      "base_traffic": 40, "rush_mult": 2.2, "night_mult": 0.15, "base_count": 150, "base_speed": 21, "lat": 30.1969, "lng": 71.4755},
    "INT_17": {"name": "Circle Chowk",           "city": "Bahawalpur",  "base_traffic": 33, "rush_mult": 2.0, "night_mult": 0.16, "base_count": 120, "base_speed": 23, "lat": 29.3956, "lng": 71.6836},
    "INT_18": {"name": "G.T. Road Chowk",        "city": "Sahiwal",     "base_traffic": 30, "rush_mult": 2.0, "night_mult": 0.17, "base_count": 115, "base_speed": 23, "lat": 30.6684, "lng": 73.1114},
    "INT_19": {"name": "Ghanta Ghar Chowk",      "city": "Dera Ghazi Khan", "base_traffic": 28, "rush_mult": 2.0, "night_mult": 0.18, "base_count": 110, "base_speed": 24, "lat": 30.0489, "lng": 70.6403},
    "INT_20": {"name": "Karsaz Chowk",           "city": "Karachi",     "base_traffic": 58, "rush_mult": 2.5, "night_mult": 0.10, "base_count": 190, "base_speed": 16, "lat": 24.8925, "lng": 67.0694},
    "INT_21": {"name": "Pacco Qillo Chowk",      "city": "Hyderabad",   "base_traffic": 44, "rush_mult": 2.3, "night_mult": 0.12, "base_count": 160, "base_speed": 20, "lat": 25.3960, "lng": 68.3570},
    "INT_22": {"name": "Barrage Chowk",          "city": "Sukkur",      "base_traffic": 32, "rush_mult": 2.1, "night_mult": 0.15, "base_count": 120, "base_speed": 22, "lat": 27.6945, "lng": 68.8367},
    "INT_23": {"name": "Doctor Chowk",           "city": "Larkana",     "base_traffic": 28, "rush_mult": 2.0, "night_mult": 0.17, "base_count": 110, "base_speed": 24, "lat": 27.5590, "lng": 68.2120},
    "INT_24": {"name": "Zarghoon Road Chowk",    "city": "Quetta",      "base_traffic": 34, "rush_mult": 2.0, "night_mult": 0.20, "base_count": 125, "base_speed": 24, "lat": 30.1790, "lng": 66.9750},
}

# Lane distribution varies by junction (deterministic, sums to 1.0)
LANE_SHARES = {
    "INT_01": {"Lane_N": 0.30, "Lane_S": 0.22, "Lane_E": 0.28, "Lane_W": 0.20},
    "INT_02": {"Lane_N": 0.35, "Lane_S": 0.15, "Lane_E": 0.30, "Lane_W": 0.20},
    "INT_03": {"Lane_N": 0.25, "Lane_S": 0.25, "Lane_E": 0.30, "Lane_W": 0.20},
    "INT_04": {"Lane_N": 0.28, "Lane_S": 0.28, "Lane_E": 0.22, "Lane_W": 0.22},
    "INT_05": {"Lane_N": 0.32, "Lane_S": 0.18, "Lane_E": 0.30, "Lane_W": 0.20},
    "INT_06": {"Lane_N": 0.26, "Lane_S": 0.24, "Lane_E": 0.28, "Lane_W": 0.22},
    "INT_07": {"Lane_N": 0.30, "Lane_S": 0.22, "Lane_E": 0.28, "Lane_W": 0.20},
    "INT_08": {"Lane_N": 0.28, "Lane_S": 0.24, "Lane_E": 0.26, "Lane_W": 0.22},
    "INT_09": {"Lane_N": 0.30, "Lane_S": 0.20, "Lane_E": 0.28, "Lane_W": 0.22},
    "INT_10": {"Lane_N": 0.34, "Lane_S": 0.20, "Lane_E": 0.26, "Lane_W": 0.20},
    "INT_11": {"Lane_N": 0.30, "Lane_S": 0.22, "Lane_E": 0.28, "Lane_W": 0.20},
    "INT_12": {"Lane_N": 0.32, "Lane_S": 0.20, "Lane_E": 0.26, "Lane_W": 0.22},
    "INT_13": {"Lane_N": 0.28, "Lane_S": 0.24, "Lane_E": 0.26, "Lane_W": 0.22},
    "INT_14": {"Lane_N": 0.30, "Lane_S": 0.20, "Lane_E": 0.28, "Lane_W": 0.22},
    "INT_15": {"Lane_N": 0.26, "Lane_S": 0.24, "Lane_E": 0.28, "Lane_W": 0.22},
    "INT_16": {"Lane_N": 0.32, "Lane_S": 0.18, "Lane_E": 0.28, "Lane_W": 0.22},
    "INT_17": {"Lane_N": 0.28, "Lane_S": 0.22, "Lane_E": 0.28, "Lane_W": 0.22},
    "INT_18": {"Lane_N": 0.30, "Lane_S": 0.22, "Lane_E": 0.26, "Lane_W": 0.22},
    "INT_19": {"Lane_N": 0.28, "Lane_S": 0.24, "Lane_E": 0.26, "Lane_W": 0.22},
    "INT_20": {"Lane_N": 0.34, "Lane_S": 0.16, "Lane_E": 0.30, "Lane_W": 0.20},
    "INT_21": {"Lane_N": 0.28, "Lane_S": 0.24, "Lane_E": 0.26, "Lane_W": 0.22},
    "INT_22": {"Lane_N": 0.30, "Lane_S": 0.22, "Lane_E": 0.26, "Lane_W": 0.22},
    "INT_23": {"Lane_N": 0.28, "Lane_S": 0.24, "Lane_E": 0.26, "Lane_W": 0.22},
    "INT_24": {"Lane_N": 0.30, "Lane_S": 0.20, "Lane_E": 0.28, "Lane_W": 0.22},
}

# Ordered node ids — the sequence defines the corridor backbone used by the
# graph forecaster for spatial correlation between nearby junctions.
NODE_IDS = list(INTERSECTIONS.keys())

DEFAULT_CSV = "pakistan_smart_traffic.csv"


# ── Live monitoring locations ────────────────────────────────────────────────
# Famous real junctions per city with GPS coordinates. These power the
# "Live Traffic" tab (TomTom queries the road at each point). The ML models
# keep using INTERSECTIONS above; this catalog is live-monitoring only.
LIVE_JUNCTIONS = {
    "Islamabad": [
        {"id": "ISB_01", "name": "Zero Point Interchange",   "lat": 33.6844, "lng": 73.0479},
        {"id": "ISB_02", "name": "Faizabad Interchange",     "lat": 33.6690, "lng": 73.0740},
        {"id": "ISB_03", "name": "Kashmir Highway 9th Ave",  "lat": 33.6870, "lng": 73.0140},
        {"id": "ISB_04", "name": "F-7 Markaz Jinnah Super",  "lat": 33.7250, "lng": 73.0470},
        {"id": "ISB_05", "name": "F-10 Markaz Chowk",        "lat": 33.6920, "lng": 72.9990},
        {"id": "ISB_06", "name": "Blue Area Jinnah Avenue",  "lat": 33.7030, "lng": 73.0510},
        {"id": "ISB_07", "name": "Peshawar Mor",             "lat": 33.6500, "lng": 73.0470},
    ],
    "Rawalpindi": [
        {"id": "RAW_01", "name": "Raja Bazaar Chowk",    "lat": 33.5658, "lng": 73.0707},
        {"id": "RAW_02", "name": "Committee Chowk",      "lat": 33.6070, "lng": 73.0670},
        {"id": "RAW_03", "name": "Mareer Chowk",         "lat": 33.6010, "lng": 73.0530},
        {"id": "RAW_04", "name": "Saddar Bank Road",     "lat": 33.5980, "lng": 73.0660},
        {"id": "RAW_05", "name": "Pir Wadhai Adda",      "lat": 33.6200, "lng": 73.0330},
        {"id": "RAW_06", "name": "Chandi Chowk",         "lat": 33.6180, "lng": 73.0780},
    ],
    "Peshawar": [
        {"id": "PES_01", "name": "Chowk Yadgar",          "lat": 34.0078, "lng": 71.5838},
        {"id": "PES_02", "name": "Hashtnagri G.T. Road",  "lat": 34.0090, "lng": 71.5610},
        {"id": "PES_03", "name": "University Road",       "lat": 34.0040, "lng": 71.4870},
        {"id": "PES_04", "name": "Karkhano Market",       "lat": 33.9970, "lng": 71.4090},
        {"id": "PES_05", "name": "Hayatabad Ring Road",   "lat": 33.9870, "lng": 71.4210},
    ],
    "Mardan": [
        {"id": "MAR_01", "name": "Kot Chowk",       "lat": 34.1980, "lng": 72.0230},
        {"id": "MAR_02", "name": "Par Hoti Chowk",  "lat": 34.1850, "lng": 72.0260},
        {"id": "MAR_03", "name": "Mardan Bypass",   "lat": 34.2080, "lng": 72.0100},
    ],
    "Swabi": [
        {"id": "SWA_01", "name": "Main Bazaar Chowk",  "lat": 34.1203, "lng": 72.4697},
        {"id": "SWA_02", "name": "Tordher Road",       "lat": 34.0830, "lng": 72.3560},
    ],
    "Abbottabad": [
        {"id": "ABB_01", "name": "Mansehra Chowk",   "lat": 34.1688, "lng": 73.2215},
        {"id": "ABB_02", "name": "Supply Bazaar",    "lat": 34.1710, "lng": 73.2170},
        {"id": "ABB_03", "name": "Mandian Chowk",    "lat": 34.1760, "lng": 73.2320},
    ],
    "Charsadda": [
        {"id": "CHA_01", "name": "Sarband Chowk",  "lat": 34.1450, "lng": 71.7310},
        {"id": "CHA_02", "name": "Tangi Road",     "lat": 34.1510, "lng": 71.7200},
    ],
    "Nowshera": [
        {"id": "NOW_01", "name": "Rashakai Chowk",    "lat": 34.0150, "lng": 71.9740},
        {"id": "NOW_02", "name": "Nowshera Cantt",    "lat": 34.0130, "lng": 71.9560},
    ],
    "Kohat": [
        {"id": "KOH_01", "name": "Stadium Chowk",  "lat": 33.5860, "lng": 71.4420},
        {"id": "KOH_02", "name": "KDA Chowk",      "lat": 33.5920, "lng": 71.4470},
    ],
    "Lahore": [
        {"id": "LHE_01", "name": "Chauburji Chowk",      "lat": 31.5685, "lng": 74.3083},
        {"id": "LHE_02", "name": "Kalma Chowk",          "lat": 31.5460, "lng": 74.3230},
        {"id": "LHE_03", "name": "Thokar Niaz Baig",     "lat": 31.5090, "lng": 74.2770},
        {"id": "LHE_04", "name": "Data Darbar",          "lat": 31.5830, "lng": 74.3100},
        {"id": "LHE_05", "name": "Shahdara Chowk",       "lat": 31.6330, "lng": 74.2870},
        {"id": "LHE_06", "name": "Mall Road",            "lat": 31.5490, "lng": 74.3080},
        {"id": "LHE_07", "name": "Liberty Market",       "lat": 31.5120, "lng": 74.3300},
        {"id": "LHE_08", "name": "Railway Station",      "lat": 31.5790, "lng": 74.3230},
    ],
    "Faisalabad": [
        {"id": "FSD_01", "name": "Clock Tower Ghanta Ghar", "lat": 31.4185, "lng": 73.0789},
        {"id": "FSD_02", "name": "Dhobi Ghat Chowk",        "lat": 31.4300, "lng": 73.0730},
        {"id": "FSD_03", "name": "Susan Road Madina Town",  "lat": 31.3970, "lng": 73.1130},
        {"id": "FSD_04", "name": "Canal Road",              "lat": 31.4350, "lng": 73.1050},
    ],
    "Gujranwala": [
        {"id": "GUJ_01", "name": "G.T. Road Bus Stand",  "lat": 32.1610, "lng": 74.1880},
        {"id": "GUJ_02", "name": "Satellite Town",       "lat": 32.1700, "lng": 74.2000},
        {"id": "GUJ_03", "name": "Model Town Chowk",     "lat": 32.1520, "lng": 74.2050},
    ],
    "Sialkot": [
        {"id": "SIA_01", "name": "Paris Road",      "lat": 32.4945, "lng": 74.5229},
        {"id": "SIA_02", "name": "Sialkot Cantt",   "lat": 32.5050, "lng": 74.5350},
        {"id": "SIA_03", "name": "Kashmir Road",    "lat": 32.4970, "lng": 74.5150},
    ],
    "Jhelum": [
        {"id": "JHE_01", "name": "G.T. Road Chowk",  "lat": 32.9424, "lng": 73.7257},
        {"id": "JHE_02", "name": "Civil Lines",      "lat": 32.9380, "lng": 73.7330},
    ],
    "Sargodha": [
        {"id": "SGD_01", "name": "Kutchery Chowk",     "lat": 32.0836, "lng": 72.6711},
        {"id": "SGD_02", "name": "University Road",    "lat": 32.0780, "lng": 72.6800},
    ],
    "Multan": [
        {"id": "MUL_01", "name": "Kumharanwala Chowk",  "lat": 30.1969, "lng": 71.4755},
        {"id": "MUL_02", "name": "Nawan Shehr Chowk",   "lat": 30.1830, "lng": 71.4910},
        {"id": "MUL_03", "name": "Gulgasht Chowk",      "lat": 30.1730, "lng": 71.4420},
        {"id": "MUL_04", "name": "Vehari Chowk",        "lat": 30.1800, "lng": 71.4870},
    ],
    "Bahawalpur": [
        {"id": "BWP_01", "name": "Circle Chowk",    "lat": 29.3956, "lng": 71.6836},
        {"id": "BWP_02", "name": "Multan Road",     "lat": 29.3900, "lng": 71.6700},
    ],
    "Sahiwal": [
        {"id": "SWL_01", "name": "G.T. Road Chowk",  "lat": 30.6684, "lng": 73.1114},
        {"id": "SWL_02", "name": "Farid Town",       "lat": 30.6620, "lng": 73.1020},
    ],
    "Dera Ghazi Khan": [
        {"id": "DGK_01", "name": "Ghanta Ghar Chowk",  "lat": 30.0489, "lng": 70.6403},
        {"id": "DGK_02", "name": "Block 20 Chowk",     "lat": 30.0550, "lng": 70.6500},
    ],
    "Karachi": [
        {"id": "KHI_01", "name": "Karsaz Chowk",           "lat": 24.8925, "lng": 67.0694},
        {"id": "KHI_02", "name": "Teen Talwar Clifton",    "lat": 24.8190, "lng": 67.0330},
        {"id": "KHI_03", "name": "Numaish Chowrangi",      "lat": 24.8670, "lng": 67.0330},
        {"id": "KHI_04", "name": "Gurumandir Chowk",       "lat": 24.8640, "lng": 67.0400},
        {"id": "KHI_05", "name": "Hassan Square Gulshan",  "lat": 24.9200, "lng": 67.1000},
        {"id": "KHI_06", "name": "Malir Halt",             "lat": 24.8930, "lng": 67.2100},
        {"id": "KHI_07", "name": "Native Jetty Bridge",    "lat": 24.8420, "lng": 67.0290},
        {"id": "KHI_08", "name": "Shahrah-e-Faisal Airport","lat": 24.9060, "lng": 67.1600},
    ],
    "Hyderabad": [
        {"id": "HYD_01", "name": "Pacco Qillo Fort",    "lat": 25.3960, "lng": 68.3570},
        {"id": "HYD_02", "name": "Auto Bhan Chowk",     "lat": 25.3790, "lng": 68.3670},
        {"id": "HYD_03", "name": "Latifabad Unit 7",    "lat": 25.3860, "lng": 68.3420},
        {"id": "HYD_04", "name": "Qasimabad Chowk",     "lat": 25.3930, "lng": 68.3330},
    ],
    "Sukkur": [
        {"id": "SUK_01", "name": "Sukkur Barrage",     "lat": 27.6945, "lng": 68.8367},
        {"id": "SUK_02", "name": "Military Road",      "lat": 27.7000, "lng": 68.8500},
    ],
    "Larkana": [
        {"id": "LAR_01", "name": "Doctor Chowk",        "lat": 27.5590, "lng": 68.2120},
        {"id": "LAR_02", "name": "Rice Canal Road",     "lat": 27.5650, "lng": 68.2200},
    ],
    "Quetta": [
        {"id": "QTA_01", "name": "Zarghoon Road",    "lat": 30.1790, "lng": 66.9750},
        {"id": "QTA_02", "name": "Sariab Road",      "lat": 30.1650, "lng": 66.9560},
        {"id": "QTA_03", "name": "Jinnah Town",      "lat": 30.2020, "lng": 66.9640},
        {"id": "QTA_04", "name": "Brewery Road",     "lat": 30.1900, "lng": 66.9830},
    ],
}


def iter_live_junctions():
    """Yield flat dicts: {id, name, city, lat, lng} for every live location."""
    for city, juncs in LIVE_JUNCTIONS.items():
        for j in juncs:
            yield {"id": j["id"], "name": j["name"], "city": city,
                   "lat": j["lat"], "lng": j["lng"]}


def live_junction_map():
    """Flat dict: junction_id -> {name, city, lat, lng}."""
    return {j["id"]: j for j in iter_live_junctions()}


def build_adjacency(node_ids=None):
    """Corridor-backbone adjacency matrix (symmetric, row normalized).

    Each junction is connected to its list neighbors (previous/next node)
    with weight 0.5 so spatially nearby junctions influence each other.
    """
    import numpy as np
    if node_ids is None:
        node_ids = NODE_IDS
    n = len(node_ids)
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        A[i, i] = 1.0
        if i > 0:
            A[i, i - 1] = 0.5
        if i < n - 1:
            A[i, i + 1] = 0.5
    A = (A + A.T) / 2.0
    row_sum = A.sum(axis=1, keepdims=True)
    A = A / np.maximum(row_sum, 1e-6)
    return A