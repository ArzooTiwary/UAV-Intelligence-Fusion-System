"""
module3_adversarial_training.py

Adversarial training experiment for Module 3.

The previous experiment (module3_real_data.py) showed that the standard
Random Forest classifier drops from 86.5% recall to 29.6% recall under
small bounded perturbations (epsilon=0.1). Verdict: FRAGILE.

This module addresses that by augmenting the training set with adversarially
perturbed spoofed samples before training. The hypothesis: a classifier
trained on both clean and perturbed spoofed examples will learn decision
boundaries that are more robust to input manipulation.

Comparison table (what we are building toward):
    Model                      AUC      Clean Recall  Adv. Recall  Drop
    Standard RF                0.9874   0.8653        0.2957       0.5696
    Standard GB                0.9923   —             —            —
    Adversarially Trained RF   TBD      TBD           TBD          TBD
"""

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler
import joblib

DATASET_PATH = "../datasets/gps_data.csv"
LABEL_COL    = "Output"
EPSILON      = 0.1   # perturbation size — same as previous experiment
N_TRIALS     = 5     # adversarial test trials


# ── LOAD AND PREPARE ─────────────────────────────────────────────
def load_and_prepare(path):
    print("Loading dataset...")
    df = pd.read_csv(path)
    df = df.fillna(df.median(numeric_only=True))

    # Same feature engineering as module3_real_data.py
    df["CN0_LC_ratio"] = df["CN0"] / (df["LC"] + 1e-6)
    df["DO_abs"]       = df["DO"].abs()
    df["PD_TCD_diff"]  = (df["PD"] - df["TCD"]).abs()
    df["CP_PD_ratio"]  = df["CP"] / (df["PD"].abs() + 1e-6)

    feature_cols = [c for c in df.columns if c != LABEL_COL]
    X = df[feature_cols]
    y = (df[LABEL_COL] > 0).astype(int)  # binary: 0=clean, 1=any spoof

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set:     {len(X_test):,} samples\n")

    return X_train_s, X_test_s, y_train, y_test, scaler, feature_cols


# ── ADVERSARIAL AUGMENTATION ──────────────────────────────────────
def augment_with_adversarial(X_train_s, y_train, epsilon, multiplier=2):
    """
    Generate adversarial examples from spoofed training samples
    and add them back into the training set.

    multiplier: how many perturbed copies to add per original spoofed sample.
    We use 2 — adding too many risks making the classifier overfit
    to the perturbation distribution rather than the underlying signal.

    This is the core of adversarial training: expose the classifier
    to the kinds of inputs an attacker might craft, so it learns
    boundaries that are harder to cross.
    """
    rng = np.random.default_rng(42)
    stds = X_train_s.std(axis=0)

    spoofed_mask = (y_train.values == 1)
    X_spoof = X_train_s[spoofed_mask]

    augmented_X = [X_train_s]
    augmented_y = [y_train.values]

    for i in range(multiplier):
        perturb = rng.uniform(-1, 1, X_spoof.shape) * stds * epsilon
        X_adv   = X_spoof + perturb
        augmented_X.append(X_adv)
        augmented_y.append(np.ones(len(X_adv), dtype=int))

    X_aug = np.vstack(augmented_X)
    y_aug = np.concatenate(augmented_y)

    # Shuffle so adversarial examples aren't all at the end
    idx = rng.permutation(len(X_aug))
    print(f"Augmented training set: {len(X_aug):,} samples "
          f"({len(X_aug) - len(X_train_s):,} adversarial examples added)")

    return X_aug[idx], y_aug[idx]


# ── TRAIN ─────────────────────────────────────────────────────────
def train_model(X_train, y_train, model_type="rf"):
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=4,
            learning_rate=0.08, random_state=42
        )
    model.fit(X_train, y_train)
    return model


# ── EVALUATE ──────────────────────────────────────────────────────
def evaluate(model, X_test_s, y_test, label="Model"):
    probs = model.predict_proba(X_test_s)[:, 1]
    preds = (probs > 0.5).astype(int)
    auc   = roc_auc_score(y_test, probs)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    print(f"\n{'='*55}")
    print(f"{label}")
    print(f"{'='*55}")
    print(classification_report(y_test, preds,
          target_names=["Authentic", "Spoofed"]))
    print(f"ROC-AUC:             {auc:.4f}")
    print(f"True Positive Rate:  {tpr:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")

    return auc, tpr, fpr, probs


# ── ADVERSARIAL ROBUSTNESS TEST ───────────────────────────────────
def adversarial_test(model, X_test_s, y_test,
                     epsilon=EPSILON, n_trials=N_TRIALS):
    spoofed_mask = (y_test.values == 1)
    X_sp  = X_test_s[spoofed_mask]
    base  = (model.predict_proba(X_sp)[:, 1] > 0.5).mean()

    rng  = np.random.default_rng(0)
    stds = X_test_s.std(axis=0)
    adv  = []
    for _ in range(n_trials):
        pert = rng.uniform(-1, 1, X_sp.shape) * stds * epsilon
        adv.append((model.predict_proba(X_sp + pert)[:, 1] > 0.5).mean())

    mean_adv = np.mean(adv)
    drop     = base - mean_adv
    verdict  = "ROBUST" if drop < 0.05 else "MODERATE" if drop < 0.15 else "FRAGILE"

    print(f"  Clean recall:          {base:.4f}")
    print(f"  Recall under attack:   {mean_adv:.4f}")
    print(f"  Drop:                  {drop:.4f}  [{verdict}]")

    return base, mean_adv, drop, verdict


# ── COMPARISON PLOTS ──────────────────────────────────────────────
def make_comparison_plots(results, y_test,
                          probs_std, probs_adv):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_color("#21262d")
        ax.tick_params(colors="#8b949e")

    # Plot 1: ROC comparison
    for probs, label, color in [
        (probs_std, "Standard RF", "#484f58"),
        (probs_adv, "Adversarially Trained RF", "#58a6ff"),
    ]:
        fpr_c, tpr_c, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        axes[0].plot(fpr_c, tpr_c, color=color,
                     label=f"{label} (AUC={auc:.3f})", lw=2)
    axes[0].plot([0,1],[0,1], color="#21262d", lw=1, ls="--")
    axes[0].set_xlabel("False Positive Rate", color="#8b949e")
    axes[0].set_ylabel("True Positive Rate", color="#8b949e")
    axes[0].set_title("ROC Curve Comparison\nStandard vs Adversarially Trained",
                      color="#c9d1d9", fontsize=11)
    axes[0].legend(facecolor="#161b22", edgecolor="#21262d",
                   labelcolor="#c9d1d9", fontsize=9)

    # Plot 2: Recall comparison bar chart
    models  = ["Standard RF", "Adversarially\nTrained RF"]
    clean   = [results["standard"]["clean_recall"],
               results["adversarial"]["clean_recall"]]
    adv_rec = [results["standard"]["adv_recall"],
               results["adversarial"]["adv_recall"]]

    x = np.arange(len(models))
    w = 0.3
    axes[1].bar(x - w/2, clean,   w, label="Clean recall",
                color="#3fb950", alpha=0.9)
    axes[1].bar(x + w/2, adv_rec, w, label="Recall under attack",
                color="#f85149", alpha=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, color="#8b949e")
    axes[1].set_ylabel("Recall", color="#8b949e")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Adversarial Robustness Comparison\n(epsilon=0.1)",
                      color="#c9d1d9", fontsize=11)
    axes[1].legend(facecolor="#161b22", edgecolor="#21262d",
                   labelcolor="#c9d1d9", fontsize=9)
    axes[1].axhline(y=0.75, color="#e3b341", lw=1, ls="--",
                    alpha=0.5, label="Target threshold")

    # Annotate bars with values
    for i, (c, a) in enumerate(zip(clean, adv_rec)):
        axes[1].text(i - w/2, c + 0.02, f"{c:.3f}",
                     ha="center", color="#3fb950", fontsize=9)
        axes[1].text(i + w/2, a + 0.02, f"{a:.3f}",
                     ha="center", color="#f85149", fontsize=9)

    plt.tight_layout()
    out = "../results_adversarial_training.png"
    plt.savefig(out, dpi=150, facecolor="#0d1117", bbox_inches="tight")
    print(f"\nComparison plot saved to {out}")


# ── MAIN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train_s, X_test_s, y_train, y_test, scaler, feature_cols = \
        load_and_prepare(DATASET_PATH)

    # ── Standard RF (baseline, reproduce previous result) ──
    print("Training standard RF (baseline)...")
    rf_std = train_model(X_train_s, y_train, "rf")
    auc_std, tpr_std, fpr_std, probs_std = evaluate(
        rf_std, X_test_s, y_test, "STANDARD RANDOM FOREST (Baseline)"
    )
    print("\nAdversarial robustness:")
    base_std, adv_std, drop_std, verdict_std = adversarial_test(
        rf_std, X_test_s, y_test
    )

    # ── Adversarially augmented training set ──
    print("\n" + "="*55)
    print("Generating adversarial training examples...")
    X_aug, y_aug = augment_with_adversarial(X_train_s, y_train, EPSILON)

    # ── Adversarially trained RF ──
    print("\nTraining adversarially augmented RF...")
    rf_adv = train_model(X_aug, y_aug, "rf")
    auc_adv, tpr_adv, fpr_adv, probs_adv = evaluate(
        rf_adv, X_test_s, y_test, "ADVERSARIALLY TRAINED RANDOM FOREST"
    )
    print("\nAdversarial robustness:")
    base_adv, adv_adv, drop_adv, verdict_adv = adversarial_test(
        rf_adv, X_test_s, y_test
    )

    # ── Save adversarially trained model ──
    joblib.dump({"model": rf_adv, "scaler": scaler,
                 "features": feature_cols, "mode": "binary_adversarial"},
                "spoof_classifier_adversarial.joblib")

    # ── Final comparison table ──
    results = {
        "standard":   {"auc": round(auc_std,4), "fpr": round(fpr_std,4),
                       "clean_recall": round(base_std,4),
                       "adv_recall": round(adv_std,4),
                       "drop": round(drop_std,4), "verdict": verdict_std},
        "adversarial":{"auc": round(auc_adv,4), "fpr": round(fpr_adv,4),
                       "clean_recall": round(base_adv,4),
                       "adv_recall": round(adv_adv,4),
                       "drop": round(drop_adv,4), "verdict": verdict_adv},
    }

    print("\n" + "="*55)
    print("FINAL COMPARISON TABLE")
    print("="*55)
    print(f"{'Metric':<28} {'Standard RF':>14} {'Adversarial RF':>16}")
    print("-"*60)
    print(f"{'ROC-AUC':<28} {results['standard']['auc']:>14} {results['adversarial']['auc']:>16}")
    print(f"{'False Positive Rate':<28} {results['standard']['fpr']:>14} {results['adversarial']['fpr']:>16}")
    print(f"{'Clean Recall':<28} {results['standard']['clean_recall']:>14} {results['adversarial']['clean_recall']:>16}")
    print(f"{'Recall Under Attack':<28} {results['standard']['adv_recall']:>14} {results['adversarial']['adv_recall']:>16}")
    print(f"{'Recall Drop':<28} {results['standard']['drop']:>14} {results['adversarial']['drop']:>16}")
    print(f"{'Verdict':<28} {results['standard']['verdict']:>14} {results['adversarial']['verdict']:>16}")

    make_comparison_plots(results, y_test, probs_std, probs_adv)
    print(json.dumps(results, indent=2))