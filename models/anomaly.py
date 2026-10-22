"""
Anomaly / Congestion-Outbreak Detector.

Flags abnormal congestion by comparing current or forecast traffic against the
historically-expected level for the same time slot:

    z = (value - expected) / std

z >= 2.0  -> WARNING   (unusually heavy)
z >= 3.0  -> CRITICAL  (outbreak - likely incident / signal failure)

Used by the backend to emit live AI alerts across the road network.
"""

SEVERITY_THRESHOLDS = {"LOW": 0.0, "WARNING": 2.0, "CRITICAL": 3.0}


def severity_from_z(z_score: float) -> str:
    if z_score >= SEVERITY_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if z_score >= SEVERITY_THRESHOLDS["WARNING"]:
        return "WARNING"
    return "LOW"


class CongestionAnomalyDetector:
    """
    Compares observed/forecast values against historical slot statistics.

    stats: dict key (int_id, lane_id, hour, minute) -> (mean_count, mean_speed)
    """

    def __init__(self, stats: dict, z_warning: float = 2.0, z_critical: float = 3.0):
        self.stats = stats
        self.z_warning = z_warning
        self.z_critical = z_critical

    def _baseline(self, int_id, lane_id, hour, minute):
        rounded = (minute // 5) * 5
        return self.stats.get((int_id, lane_id, hour, rounded))

    def detect_lane(self, int_id, lane_id, hour, minute, count, speed):
        """
        Returns (severity, z_score, message) for one lane observation.
        """
        base = self._baseline(int_id, lane_id, hour, minute)
        if base is None or base[0] <= 0:
            return ("LOW", 0.0, f"{lane_id}: no historical baseline")

        mean_count, mean_speed = base
        std = max(mean_count * 0.25, 2.0)  # empirical std proxy
        z = (count - mean_count) / std
        severity = "LOW"
        if z >= self.z_critical:
            severity = "CRITICAL"
        elif z >= self.z_warning:
            severity = "WARNING"

        if severity == "LOW":
            return (severity, z, f"{lane_id}: traffic close to expected ({int(count)} vs {int(mean_count)})")

        if severity == "CRITICAL":
            msg = (f"{lane_id}: CONGESTION OUTBREAK - {int(count)} vehicles "
                   f"(expected {int(mean_count)}), z={z:.1f}")
        else:
            msg = (f"{lane_id}: unusually heavy - {int(count)} vehicles "
                   f"(expected {int(mean_count)}), z={z:.1f}")
        return (severity, z, msg)