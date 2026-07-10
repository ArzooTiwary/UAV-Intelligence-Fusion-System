import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────
# WHAT IS THIS MODULE?
# ─────────────────────────────────────────
# Module 1 tells us what's in the image.
# Module 2 tells us how suspicious the GPS is.
# Module 3 tells us if it looks like spoofing.
#
# But none of them talk to each other.
# Module 4 combines all three into one threat score.
#
# We use Dempster-Shafer evidence theory.
# The idea: each source gives a "belief" between 0 and 1.
# We combine beliefs using a mathematical rule.
# The result is one final threat score with a confidence range.
#
# Why not just add points (+10 per vehicle, +40 for spoofing)?
# Because that ignores uncertainty. What if the camera is foggy
# and detections are unreliable? DS theory handles that naturally
# by letting each source say "I'm not sure" instead of forcing
# a hard number.
# ─────────────────────────────────────────


def vision_belief(vehicle_count, person_count, loitering):
    """
    Convert vision detections into a threat belief score.

    More vehicles + loitering = higher belief in threat.
    But vision alone is never fully certain (camera angle,
    weather, occlusion) so we cap belief below 1.0 and keep
    meaningful uncertainty mass.

    Returns: (belief_threat, belief_benign, uncertainty)
    These three must always add up to exactly 1.0.
    """
    # Score based on what was detected
    activity = (vehicle_count * 1.0) + (person_count * 0.5) + (3.0 if loitering else 0.0)

    # Convert to a belief score, capped at 0.75
    # Cap exists because vision alone can never fully confirm a threat
    belief_threat = min(0.75, activity / 15.0)

    # We always keep 25% uncertainty from vision
    # (fog, angle, small image size can fool the detector)
    uncertainty = 0.25

    belief_benign = max(0.0, 1.0 - belief_threat - uncertainty)

    return belief_threat, belief_benign, uncertainty


def cyber_belief(spoof_confidence):
    """
    Convert Module 3's spoofing confidence into a belief score.

    The classifier already gives us a probability (0 to 1).
    We scale it slightly down because no classifier is perfect,
    keeping a small uncertainty buffer.
    """
    belief_threat = spoof_confidence * 0.90
    uncertainty = 0.10
    belief_benign = max(0.0, 1.0 - belief_threat - uncertainty)

    return belief_threat, belief_benign, uncertainty


def context_belief(in_restricted_zone, is_nighttime):
    """
    Context doesn't detect anything itself.
    It shifts how much weight we give to the other sources.

    Example: 5 vehicles at a checkpoint = normal.
             5 vehicles in a restricted zone at night = suspicious.
    Same vision evidence, very different threat meaning.
    """
    if in_restricted_zone and is_nighttime:
        return 0.55, 0.05, 0.40
    elif in_restricted_zone:
        return 0.35, 0.15, 0.50
    elif is_nighttime:
        return 0.20, 0.30, 0.50
    else:
        return 0.05, 0.55, 0.40


def dempster_combine(source1, source2):
    """
    Dempster's rule of combination.

    Takes two evidence sources, each as (belief_threat, belief_benign, uncertainty).
    Returns combined (belief_threat, belief_benign, uncertainty).

    The math:
    - Multiply every combination of mass assignments
    - When two sources agree (both say threat OR both say benign) → add to combined belief
    - When two sources disagree (one says threat, other says benign) → this is "conflict"
    - Conflict mass gets discarded and the rest gets renormalized to sum to 1.0

    Why renormalize? Because Dempster said: if two reliable sources disagree,
    the disagreement itself is evidence that one source might be wrong —
    so we redistribute that probability rather than letting it inflate uncertainty.
    """
    m1_t, m1_b, m1_u = source1   # threat, benign, uncertainty for source 1
    m2_t, m2_b, m2_u = source2   # threat, benign, uncertainty for source 2

    # Agreement masses
    agree_threat = m1_t * m2_t + m1_t * m2_u + m1_u * m2_t
    agree_benign = m1_b * m2_b + m1_b * m2_u + m1_u * m2_b
    agree_unknown = m1_u * m2_u

    # Conflict = one says threat, other says benign
    conflict = m1_t * m2_b + m1_b * m2_t

    # Renormalize (discard conflict mass)
    norm = 1.0 - conflict

    if norm < 1e-9:
        # Sources completely contradict each other
        # Fall back to maximum uncertainty rather than crash
        return 0.0, 0.0, 1.0

    combined_threat = agree_threat / norm
    combined_benign = agree_benign / norm
    combined_unknown = agree_unknown / norm

    return combined_threat, combined_benign, combined_unknown


def fuse(vehicle_count, person_count, loitering,
         spoof_confidence, in_restricted_zone, is_nighttime):
    """
    Main fusion function.
    Takes inputs from all three modules + context.
    Returns one unified threat assessment.
    """

    # Get belief from each source
    v = vision_belief(vehicle_count, person_count, loitering)
    c = cyber_belief(spoof_confidence)
    ctx = context_belief(in_restricted_zone, is_nighttime)

    # Combine vision + cyber first
    fused_vc = dempster_combine(v, c)

    # Then combine result with context
    fused_final = dempster_combine(fused_vc, ctx)

    belief_threat, belief_benign, uncertainty = fused_final

    # Plausibility = maximum possible threat belief
    # = threat belief + uncertainty (uncertainty could all be threat)
    plausibility = belief_threat + uncertainty

    # Classify threat level
    if belief_threat >= 0.65:
        level = "HIGH"
    elif belief_threat >= 0.35:
        level = "MEDIUM"
    else:
        level = "LOW"

    # Build list of what drove this assessment
    factors = []
    if vehicle_count >= 3:
        factors.append("multi_vehicle_cluster")
    if loitering:
        factors.append("loitering_detected")
    if spoof_confidence > 0.5:
        factors.append("gps_spoofing_likely")
    if in_restricted_zone:
        factors.append("restricted_zone")
    if is_nighttime:
        factors.append("nighttime_activity")
    if not factors:
        factors.append("no_significant_indicators")

    return {
        "threat_level": level,
        "threat_score": round(belief_threat, 3),
        "confidence_range": [round(belief_threat, 3), round(plausibility, 3)],
        "contributing_factors": factors,
        "evidence_breakdown": {
            "vision":  {"threat": round(v[0], 3), "benign": round(v[1], 3), "uncertain": round(v[2], 3)},
            "cyber":   {"threat": round(c[0], 3), "benign": round(c[1], 3), "uncertain": round(c[2], 3)},
            "context": {"threat": round(ctx[0], 3), "benign": round(ctx[1], 3), "uncertain": round(ctx[2], 3)},
        }
    }


if __name__ == "__main__":

    print("=" * 55)
    print("SCENARIO A: Normal patrol area, daytime, no anomaly")
    print("=" * 55)
    result = fuse(
        vehicle_count=2,
        person_count=1,
        loitering=False,
        spoof_confidence=0.05,
        in_restricted_zone=False,
        is_nighttime=False
    )
    print(json.dumps(result, indent=2))

    print()
    print("=" * 55)
    print("SCENARIO B: Restricted zone, night, GPS spoofed, loitering")
    print("=" * 55)
    result = fuse(
        vehicle_count=5,
        person_count=2,
        loitering=True,
        spoof_confidence=0.91,
        in_restricted_zone=True,
        is_nighttime=True
    )
    print(json.dumps(result, indent=2))

    print()
    print("=" * 55)
    print("SCENARIO C: Vehicles spotted but NO cyber anomaly")
    print("=" * 55)
    result = fuse(
        vehicle_count=4,
        person_count=0,
        loitering=False,
        spoof_confidence=0.08,
        in_restricted_zone=True,
        is_nighttime=False
    )
    print(json.dumps(result, indent=2))