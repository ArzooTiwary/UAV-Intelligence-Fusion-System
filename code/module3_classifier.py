import numpy as np
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from module2_ekf import UAVKalmanFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ─────────────────────────────────────────
# STEP 1: Generate training data
# ─────────────────────────────────────────
# In a real project you'd load a real GPS spoofing dataset here.
# For now we simulate clean flights and spoofed flights so we
# have something to train and test on.

def generate_flight(n_steps=100, spoofed=False, seed=42):
    """
    Simulate one UAV flight and extract features from it.
    Returns a list of feature rows, each with a label (0=clean, 1=spoofed).
    """
    rng = np.random.default_rng(seed)
    ekf = UAVKalmanFilter(dt=0.5)

    # Starting position
    lat = 28.6139
    lon = 77.2090
    alt = 400.0

    rows = []
    prev_speed = 18.0
    prev_alt = alt
    prev_heading = 90.0

    # If spoofed, pick a random step to inject the attack
    spoof_step = rng.integers(30, 70) if spoofed else -1

    for i in range(n_steps):
        # Normal smooth movement
        lat += rng.normal(0.00001, 0.000005)
        lon += rng.normal(0.00001, 0.000005)
        alt += rng.normal(0, 0.5)
        speed = 18.0 + rng.normal(0, 0.5)
        heading = 90.0 + rng.normal(0, 1.0)

        label = 0

        # Inject spoofing attack at spoof_step
        if spoofed and i == spoof_step:
            lat += rng.uniform(0.005, 0.015)   # sudden position jump
            lon += rng.uniform(0.005, 0.015)
            alt += rng.uniform(20, 60)          # sudden altitude jump
            speed = rng.uniform(1, 5)           # speed drops (spoofed signal looks too smooth)
            heading = rng.uniform(0, 360)       # heading jumps randomly
            label = 1

        # Get EKF residual
        result = ekf.step(lat, lon, alt, speed, heading)
        residual = result["innovation_residual"]

        # Build feature row
        # These are the signals the classifier looks at together
        speed_change = abs(speed - prev_speed)
        alt_change = abs(alt - prev_alt)
        heading_change = abs(heading - prev_heading)
        if heading_change > 180:
            heading_change = 360 - heading_change  # handle wrap-around (359 -> 1 is only 2 degrees)

        rows.append({
            "residual": residual,
            "speed": speed,
            "speed_change": speed_change,
            "alt_change": alt_change,
            "heading_change": heading_change,
            "label": label
        })

        prev_speed = speed
        prev_alt = alt
        prev_heading = heading

    return rows


def build_dataset(n_clean=60, n_spoofed=60):
    all_rows = []

    for i in range(n_clean):
        rows = generate_flight(n_steps=100, spoofed=False, seed=i)
        all_rows.extend(rows)

    for i in range(n_spoofed):
        rows = generate_flight(n_steps=100, spoofed=True, seed=1000 + i)
        all_rows.extend(rows)

    return all_rows


# ─────────────────────────────────────────
# STEP 2: Train the classifier
# ─────────────────────────────────────────

def train_classifier(dataset):
    feature_cols = ["residual", "speed", "speed_change", "alt_change", "heading_change"]

    X = np.array([[row[col] for col in feature_cols] for row in dataset])
    y = np.array([row["label"] for row in dataset])

    # Split: 70% for training, 30% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train the Random Forest
    # n_estimators=100 means 100 decision trees vote together
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("=== Classifier Performance ===")
    print(classification_report(y_test, y_pred, target_names=["clean", "spoofed"]))

    return model, feature_cols


# ─────────────────────────────────────────
# STEP 3: Use the classifier on new data
# ─────────────────────────────────────────

def classify_reading(model, feature_cols, residual, speed, speed_change,
                     alt_change, heading_change):
    features = np.array([[residual, speed, speed_change, alt_change, heading_change]])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][prediction]

    return {
        "status": "spoofed" if prediction == 1 else "clean",
        "confidence": round(float(confidence), 3),
        "features_used": {
            "residual": round(residual, 4),
            "speed": round(speed, 2),
            "speed_change": round(speed_change, 2),
            "alt_change": round(alt_change, 2),
            "heading_change": round(heading_change, 2)
        }
    }


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Building training dataset...")
    dataset = build_dataset()
    print(f"Dataset size: {len(dataset)} readings")
    spoofed_count = sum(1 for r in dataset if r["label"] == 1)
    print(f"Spoofed: {spoofed_count}, Clean: {len(dataset) - spoofed_count}\n")

    print("Training classifier...")
    model, feature_cols = train_classifier(dataset)

    # Test on a clearly clean reading
    print("\n=== Test: Clean reading ===")
    result = classify_reading(model, feature_cols,
                               residual=0.65,
                               speed=18.1,
                               speed_change=0.1,
                               alt_change=0.3,
                               heading_change=0.5)
    print(json.dumps(result, indent=2))

    # Test on a clearly spoofed reading
    print("\n=== Test: Spoofed reading ===")
    result = classify_reading(model, feature_cols,
                               residual=112.6,
                               speed=3.2,
                               speed_change=14.8,
                               alt_change=50.0,
                               heading_change=95.0)
    print(json.dumps(result, indent=2))