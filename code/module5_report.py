import json
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────
# WHAT IS THIS MODULE?
# ─────────────────────────────────────────
# Module 4 gives us structured JSON.
# Module 5 turns that JSON into a plain English report
# that any operator can read and act on.
#
# IMPORTANT RULE: Every sentence must come directly
# from the JSON. We never invent or assume anything.
# This is what makes AI reporting safe in a real system.
# ─────────────────────────────────────────


def generate_report(fusion_result, vehicle_count=0, person_count=0,
                    location="Unknown Sector", timestamp=None):
    """
    Takes the output of Module 4's fuse() function and
    converts it into a plain English intelligence report.

    Every sentence traces back to a specific field in fusion_result.
    Nothing is invented.
    """

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    level = fusion_result["threat_level"]
    score = fusion_result["threat_score"]
    low, high = fusion_result["confidence_range"]
    factors = fusion_result["contributing_factors"]

    # Pick threat level colour word for the summary line
    level_word = {
        "HIGH":   "HIGH — immediate attention required",
        "MEDIUM": "MEDIUM — monitor closely",
        "LOW":    "LOW — situation appears normal"
    }[level]

    sentences = []

    # ── Detection summary ──
    if vehicle_count > 0 or person_count > 0:
        parts = []
        if vehicle_count > 0:
            parts.append(f"{vehicle_count} vehicle(s)")
        if person_count > 0:
            parts.append(f"{person_count} person(s)")
        sentences.append(
            f"{' and '.join(parts)} detected in {location}."
        )
    else:
        sentences.append(f"No significant objects detected in {location}.")

    # ── Behavioural observations ──
    if "loitering_detected" in factors:
        sentences.append(
            "At least one track showed loitering behaviour — "
            "stationary or slow-moving for an extended period."
        )

    if "multi_vehicle_cluster" in factors:
        sentences.append(
            "Multiple vehicles detected in close proximity, "
            "consistent with a vehicle cluster or convoy."
        )

    # ── Cyber / telemetry findings ──
    if "gps_spoofing_likely" in factors:
        sentences.append(
            "Telemetry analysis flagged a GPS anomaly. "
            "Innovation residual exceeded normal thresholds, "
            "consistent with GPS spoofing or signal manipulation."
        )
    else:
        sentences.append(
            "No telemetry anomalies detected. "
            "GPS signal appears normal."
        )

    # ── Context ──
    context_parts = []
    if "restricted_zone" in factors:
        context_parts.append(f"{location} is designated a restricted zone")
    if "nighttime_activity" in factors:
        context_parts.append("activity occurred during nighttime hours")

    if context_parts:
        sentences.append(
            f"Note: {'; '.join(context_parts)}."
        )

    # ── Final assessment ──
    sentences.append(
        f"Fused threat assessment: {level_word}. "
        f"Threat score: {score} "
        f"(confidence range {low} – {high})."
    )

    # ── Build full report ──
    report_text = " ".join(sentences)

    report = {
        "timestamp": timestamp,
        "location": location,
        "threat_level": level,
        "threat_score": score,
        "report": report_text,
        "contributing_factors": factors,
        "confidence_range": fusion_result["confidence_range"]
    }

    return report


def save_report(report, output_path="last_report.json"):
    """Save the report to a JSON file so other modules can read it."""
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {output_path}")


def print_report(report):
    """Print a clean, readable version to the terminal."""
    print()
    print("=" * 60)
    print("       UAV INTELLIGENCE FUSION SYSTEM — REPORT")
    print("=" * 60)
    print(f"  Timestamp : {report['timestamp']}")
    print(f"  Location  : {report['location']}")
    print(f"  Threat    : {report['threat_level']}  (score: {report['threat_score']})")
    print("=" * 60)
    print()
    print(report["report"])
    print()
    print(f"  Factors   : {', '.join(report['contributing_factors'])}")
    print("=" * 60)
    print()


# ─────────────────────────────────────────
# TEST IT
# ─────────────────────────────────────────

if __name__ == "__main__":
    from module4_fusion import fuse

    print("\n>>> SCENARIO A: Normal patrol, no anomaly")
    result_a = fuse(
        vehicle_count=2, person_count=1, loitering=False,
        spoof_confidence=0.05,
        in_restricted_zone=False, is_nighttime=False
    )
    report_a = generate_report(
        result_a, vehicle_count=2, person_count=1,
        location="Sector 2"
    )
    print_report(report_a)

    print("\n>>> SCENARIO B: Restricted zone, night, GPS spoofed")
    result_b = fuse(
        vehicle_count=5, person_count=2, loitering=True,
        spoof_confidence=0.91,
        in_restricted_zone=True, is_nighttime=True
    )
    report_b = generate_report(
        result_b, vehicle_count=5, person_count=2,
        location="Sector 7 — Restricted"
    )
    print_report(report_b)
    save_report(report_b)