"""
pipeline.py — Full end-to-end UAV-IFS pipeline

Connects all six modules into a single callable function.
Input:  image path + one telemetry reading
Output: full threat assessment with report

This is the file that proves the system works end-to-end,
not just as six independent demos.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module2_ekf import UAVKalmanFilter
from module4_fusion import fuse
from module5_report import generate_report


# ── MODULE 1: VISION ─────────────────────────────────────────────
def run_vision(image_path, conf_threshold=0.4):
    """
    Run YOLOv8 on an image and return structured counts.
    Falls back to zero counts if image path is None or
    ultralytics is not available.
    """
    if image_path is None:
        return {"vehicles": 0, "people": 0, "boats": 0}

    try:
        from ultralytics import YOLO
        VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
        PEOPLE_CLASSES  = {"person"}
        BOAT_CLASSES    = {"boat"}

        model   = YOLO("yolov8n.pt")
        results = model.predict(source=image_path,
                                conf=conf_threshold, verbose=False)
        result  = results[0]
        counts  = {"vehicles": 0, "people": 0, "boats": 0}

        if result.boxes is not None:
            for cls_id in result.boxes.cls.cpu().numpy().astype(int):
                name = result.names[cls_id]
                if name in VEHICLE_CLASSES:
                    counts["vehicles"] += 1
                elif name in PEOPLE_CLASSES:
                    counts["people"]   += 1
                elif name in BOAT_CLASSES:
                    counts["boats"]    += 1
        return counts

    except Exception as e:
        print(f"[vision] warning: {e}")
        return {"vehicles": 0, "people": 0, "boats": 0}


# ── MODULE 2+3: TELEMETRY → EKF → CLASSIFIER ─────────────────────
# We keep one EKF instance alive across calls so it builds
# a running state estimate rather than starting fresh each time
_ekf = UAVKalmanFilter(dt=0.5)


def run_cyber(lat, lon, alt, speed, heading,
              classifier_path="spoof_classifier_adversarial.joblib"):
    """
    Run one telemetry reading through the EKF and then the
    adversarially trained spoofing classifier.
    Returns spoof probability (0.0 to 1.0).
    """
    global _ekf

    # Module 2 — EKF
    ekf_result  = _ekf.step(lat, lon, alt, speed, heading)
    mahalanobis = ekf_result["innovation_residual"]

    # Module 3 — classifier
    try:
        import joblib
        bundle   = joblib.load(classifier_path)
        model    = bundle["model"]
        scaler   = bundle["scaler"]
        features = bundle["features"]

        # Build feature row matching the real dataset columns
        # For live telemetry we use what we have; unknown
        # GPS signal features (PRN, DO, etc.) default to 0
        row = {f: 0.0 for f in features}

        # Fill in what we actually have
        row["CN0"]          = alt          # altitude as CN0 proxy
        row["LC"]           = speed        # speed as LC proxy
        row["DO"]           = heading      # heading as Doppler proxy
        row["TCD"]          = mahalanobis  # EKF residual as TCD proxy
        row["CN0_LC_ratio"] = alt / (speed + 1e-6)
        row["DO_abs"]       = abs(heading)
        row["PD_TCD_diff"]  = mahalanobis
        row["CP_PD_ratio"]  = 0.0

        X = pd.DataFrame([row])[features]
        X_s = scaler.transform(X)
        spoof_prob = float(model.predict_proba(X_s)[0][1])

    except Exception as e:
        print(f"[cyber] warning: {e} — using EKF residual heuristic")
        # Fallback: convert Mahalanobis residual to a 0-1 probability
        spoof_prob = float(min(1.0, mahalanobis / 50.0))

    return spoof_prob, mahalanobis


# ── FULL PIPELINE ─────────────────────────────────────────────────
def run_pipeline(
    image_path=None,
    lat=28.6139, lon=77.2090, alt=400.0,
    speed=18.0,  heading=90.0,
    in_restricted_zone=False,
    is_nighttime=False,
    location="Unknown Sector",
    verbose=True
):
    """
    Run the complete UAV-IFS pipeline on one frame.

    Parameters
    ----------
    image_path        : path to aerial image (None = skip vision)
    lat/lon/alt       : GPS reading
    speed/heading     : telemetry
    in_restricted_zone: context flag
    is_nighttime      : context flag
    location          : sector label for the report
    verbose           : print intermediate outputs

    Returns
    -------
    dict with full threat assessment and report
    """

    # Module 1
    if verbose:
        print("\n[Module 1] Running vision detection...")
    vision = run_vision(image_path)
    if verbose:
        print(f"           Detections: {vision}")

    # Modules 2 + 3
    if verbose:
        print("[Module 2+3] Running EKF + cyber anomaly detection...")
    spoof_prob, residual = run_cyber(lat, lon, alt, speed, heading)
    if verbose:
        print(f"           EKF residual:    {residual:.4f}")
        print(f"           Spoof probability: {spoof_prob:.4f}")

    # Module 4
    if verbose:
        print("[Module 4] Running Dempster-Shafer fusion...")
    loitering = False   # set True if tracker detects loitering
    fusion = fuse(
        vehicle_count=vision["vehicles"],
        person_count=vision["people"],
        loitering=loitering,
        spoof_confidence=spoof_prob,
        in_restricted_zone=in_restricted_zone,
        is_nighttime=is_nighttime
    )
    if verbose:
        print(f"           Threat level: {fusion['threat_level']}  "
              f"(score: {fusion['threat_score']})")

    # Module 5
    if verbose:
        print("[Module 5] Generating intelligence report...")
    report = generate_report(
        fusion,
        vehicle_count=vision["vehicles"],
        person_count=vision["people"],
        location=location
    )

    output = {
        "location":     location,
        "vision":       vision,
        "ekf_residual": round(residual, 4),
        "spoof_prob":   round(spoof_prob, 4),
        "fusion":       fusion,
        "report":       report["report"],
        "timestamp":    report["timestamp"],
    }

    return output


# ── TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("UAV-IFS END-TO-END PIPELINE TEST")
    print("=" * 60)

    # Test 1 — no image, normal telemetry
    print("\n--- Test 1: Normal flight, no anomaly ---")
    result = run_pipeline(
        image_path=None,
        lat=28.6139, lon=77.2090, alt=400, speed=18, heading=90,
        in_restricted_zone=False, is_nighttime=False,
        location="Sector 2"
    )
    print(f"\nThreat: {result['fusion']['threat_level']}  "
          f"Score: {result['fusion']['threat_score']}")
    print(f"Report: {result['report']}")

    # Test 2 — simulated GPS anomaly
    print("\n--- Test 2: GPS anomaly, restricted zone, nighttime ---")
    # Reset EKF for fresh test
    _ekf.__init__(dt=0.5)
    # Feed 5 normal readings first so EKF has a baseline
    for i in range(5):
        run_cyber(28.6139 + i*0.00001, 77.2090, 400, 18, 90)
    # Then inject a spoofed jump
    result2 = run_pipeline(
        image_path=None,
        lat=28.6200, lon=77.2200, alt=450, speed=3, heading=45,
        in_restricted_zone=True, is_nighttime=True,
        location="Sector 7 — Restricted"
    )
    print(f"\nThreat: {result2['fusion']['threat_level']}  "
          f"Score: {result2['fusion']['threat_score']}")
    print(f"EKF residual: {result2['ekf_residual']}")
    print(f"Report: {result2['report']}")

    # Test 3 — with real image (if bus.jpg exists)
    bus_path = "../bus.jpg" if os.path.exists("../bus.jpg") else None
    if bus_path:
        print("\n--- Test 3: Real image detection ---")
        _ekf.__init__(dt=0.5)
        result3 = run_pipeline(
            image_path=bus_path,
            lat=28.6139, lon=77.2090, alt=400, speed=18, heading=90,
            in_restricted_zone=False, is_nighttime=False,
            location="Sector 3"
        )
        print(f"\nVision: {result3['vision']}")
        print(f"Threat: {result3['fusion']['threat_level']}  "
              f"Score: {result3['fusion']['threat_score']}")
        print(f"Report: {result3['report']}")
    else:
        print("\n--- Test 3 skipped (no image file found) ---")
        print("Place any image in the project root as 'bus.jpg' to test vision")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)