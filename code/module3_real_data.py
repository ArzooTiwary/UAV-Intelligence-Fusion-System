"""
module3_real_data.py

Module 3 retrained on the real Aissou 2022 GPS spoofing dataset.

Dataset: GPS_Data_Simplified_2D_Feature_Map (Mendeley, Aissou 2022)
Columns: PRN, DO, PD, RX, TOW, CP, EC, LC, PC, PIP, PQP, TCD, CN0, Output

Output labels:
    0 = Authentic GPS signal
    1 = Spoofing attack type 1
    2 = Spoofing attack type 2
    3 = Spoofing attack type 3

We run two experiments:
    1. Binary classification  (clean vs any spoofing)
    2. Multi-class            (clean vs attack type 1/2/3)

This directly answers a more useful operational question than binary
detection alone: not just "is the signal spoofed?" but "what kind of
attack is this?"
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, roc_curve,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler, label_binarize
import joblib

DATASET_PATH = "../datasets/gps_data.csv"
LABEL_COL    = "Output"


# ── LOAD ─────────────────────────────────────────────────────────
def load_data(path):
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"\nLabel distribution:")
    vc = df[LABEL_COL].value_counts().sort_index()
    labels = {0: "Authentic", 1: "Spoof Type 1",
              2: "Spoof Type 2", 3: "Spoof Type 3"}
    for k, v in vc.items():
        print(f"  {k} ({labels.get(k,'Unknown')}): {v:,} samples ({v/len(df):.1%})")
    print()
    return df


# ── FEATURE ENGINEERING ──────────────────────────────────────────
def engineer_features(df):
    df = df.copy().fillna(df.median(numeric_only=True))

    # Physically motivated interaction features
    # CN0/LC ratio: spoofed signals often have high signal strength
    # but short lock time because the receiver just acquired them
    df["CN0_LC_ratio"]  = df["CN0"] / (df["LC"] + 1e-6)

    # Absolute Doppler: spoofed signals can have unnaturally
    # consistent or near-zero Doppler shift
    df["DO_abs"]        = df["DO"].abs()

    # Pseudorange vs clock difference inconsistency
    df["PD_TCD_diff"]   = (df["PD"] - df["TCD"]).abs()

    # Carrier phase to pseudorange ratio
    df["CP_PD_ratio"]   = df["CP"] / (df["PD"].abs() + 1e-6)

    return df


# ── EXPERIMENT 1: BINARY ─────────────────────────────────────────
def binary_experiment(df, feature_cols):
    print("=" * 55)
    print("EXPERIMENT 1 — Binary Classification")
    print("Clean (0) vs Any Spoofing Attack (1/2/3)")
    print("=" * 55)

    X = df[feature_cols]
    y = (df[LABEL_COL] > 0).astype(int)  # 0=clean, 1=any spoof

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Random Forest
    print("\nTraining Random Forest (binary)...")
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    rf_probs = rf.predict_proba(X_test_s)[:, 1]
    rf_preds = (rf_probs > 0.5).astype(int)
    rf_auc   = roc_auc_score(y_test, rf_probs)
    tn, fp, fn, tp = confusion_matrix(y_test, rf_preds).ravel()
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)

    print(classification_report(y_test, rf_preds,
          target_names=["Authentic", "Spoofed"]))
    print(f"ROC-AUC:             {rf_auc:.4f}")
    print(f"True Positive Rate:  {tpr:.4f}  (detection rate)")
    print(f"False Positive Rate: {fpr:.4f}  "
          f"(~1 false alarm per {int(1/fpr) if fpr>0 else 'inf'} clean readings)")

    # Gradient Boosting
    print("\nTraining Gradient Boosting (binary)...")
    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=4,
        learning_rate=0.08, random_state=42
    )
    gb.fit(X_train_s, y_train)
    gb_probs = gb.predict_proba(X_test_s)[:, 1]
    gb_auc   = roc_auc_score(y_test, gb_probs)
    gb_preds = (gb_probs > 0.5).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_test, gb_preds).ravel()

    print(classification_report(y_test, gb_preds,
          target_names=["Authentic", "Spoofed"]))
    print(f"ROC-AUC:             {gb_auc:.4f}")
    print(f"False Positive Rate: {fp2/(fp2+tn2):.4f}")

    return {
        "random_forest": {"roc_auc": round(rf_auc,4),
                          "tpr": round(tpr,4), "fpr": round(fpr,4)},
        "gradient_boosting": {"roc_auc": round(gb_auc,4),
                              "fpr": round(fp2/(fp2+tn2),4)},
    }, rf, gb, scaler, X_test_s, y_test, rf_probs, gb_probs


# ── EXPERIMENT 2: MULTI-CLASS ────────────────────────────────────
def multiclass_experiment(df, feature_cols):
    print("\n" + "=" * 55)
    print("EXPERIMENT 2 — Multi-Class Classification")
    print("Authentic vs Spoof Type 1 vs Type 2 vs Type 3")
    print("=" * 55)

    X = df[feature_cols]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("\nTraining Random Forest (multi-class)...")
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    preds = rf.predict(X_test_s)
    probs = rf.predict_proba(X_test_s)

    class_names = ["Authentic", "Spoof T1", "Spoof T2", "Spoof T3"]
    print(classification_report(y_test, preds, target_names=class_names))

    # One-vs-rest AUC for multi-class
    y_bin = label_binarize(y_test, classes=[0,1,2,3])
    auc_ovr = roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")
    print(f"Macro ROC-AUC (one-vs-rest): {auc_ovr:.4f}")

    return rf, scaler, X_test_s, y_test, preds, class_names, auc_ovr


# ── ADVERSARIAL ROBUSTNESS ────────────────────────────────────────
def adversarial_test(model, X_test_s, y_test_binary,
                     epsilon=0.1, n_trials=5):
    print("\n" + "=" * 55)
    print("ADVERSARIAL ROBUSTNESS TEST")
    print(f"Bounded perturbation (epsilon={epsilon})")
    print("=" * 55)

    spoofed = (y_test_binary == 1)
    X_sp    = X_test_s[spoofed]
    base    = (model.predict_proba(X_sp)[:,1] > 0.5).mean()

    rng  = np.random.default_rng(0)
    stds = X_test_s.std(axis=0)
    adv  = []
    for _ in range(n_trials):
        pert  = rng.uniform(-1,1, X_sp.shape) * stds * epsilon
        adv.append((model.predict_proba(X_sp + pert)[:,1] > 0.5).mean())

    mean_adv = np.mean(adv)
    drop     = base - mean_adv

    print(f"Baseline recall (spoofed samples): {base:.4f}")
    print(f"Recall under perturbation:         {mean_adv:.4f}")
    print(f"Recall drop:                       {drop:.4f}")

    if drop < 0.05:
        verdict = "ROBUST"
    elif drop < 0.15:
        verdict = "MODERATE"
    else:
        verdict = "FRAGILE"
    print(f"Verdict: {verdict}")

    return base, mean_adv, drop


# ── PLOTS ─────────────────────────────────────────────────────────
def make_plots(y_test_bin, rf_probs, gb_probs,
               rf_mc, y_test_mc, mc_preds, class_names,
               feature_cols, rf_importances):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_color("#21262d")
        ax.tick_params(colors="#8b949e")

    # Plot 1: ROC curves (binary)
    for probs, label, color in [
        (rf_probs, "Random Forest", "#58a6ff"),
        (gb_probs, "Gradient Boosting", "#3fb950"),
    ]:
        fpr_c, tpr_c, _ = roc_curve(y_test_bin, probs)
        auc = roc_auc_score(y_test_bin, probs)
        axes[0].plot(fpr_c, tpr_c, color=color,
                     label=f"{label} (AUC={auc:.3f})", lw=2)
    axes[0].plot([0,1],[0,1], color="#484f58", lw=1, ls="--")
    axes[0].set_xlabel("False Positive Rate", color="#8b949e")
    axes[0].set_ylabel("True Positive Rate", color="#8b949e")
    axes[0].set_title("ROC — Binary Spoofing Detection\n(Real Dataset, n=510,530)",
                      color="#c9d1d9", fontsize=10)
    axes[0].legend(facecolor="#161b22", edgecolor="#21262d",
                   labelcolor="#c9d1d9", fontsize=8)

    # Plot 2: Confusion matrix (multi-class)
    cm = confusion_matrix(y_test_mc, mc_preds)
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks(range(4))
    axes[1].set_yticks(range(4))
    axes[1].set_xticklabels(class_names, color="#8b949e", fontsize=8)
    axes[1].set_yticklabels(class_names, color="#8b949e", fontsize=8)
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f"{cm[i,j]:,}",
                         ha="center", va="center",
                         color="white" if cm[i,j] > cm.max()/2 else "#8b949e",
                         fontsize=7)
    axes[1].set_title("Confusion Matrix — Multi-Class\nAttack Type Classification",
                      color="#c9d1d9", fontsize=10)
    axes[1].set_xlabel("Predicted", color="#8b949e")
    axes[1].set_ylabel("Actual", color="#8b949e")

    # Plot 3: Feature importance
    imp = pd.Series(rf_importances, index=feature_cols).sort_values()
    top = imp.tail(12)
    axes[2].barh(top.index, top.values, color="#1f6feb")
    axes[2].set_xlabel("Importance", color="#8b949e")
    axes[2].set_title("Feature Importance\n(Random Forest, Binary)",
                      color="#c9d1d9", fontsize=10)

    plt.tight_layout()
    out = "../results_real_dataset.png"
    plt.savefig(out, dpi=150, facecolor="#0d1117", bbox_inches="tight")
    print(f"\nPlots saved to {out}")


# ── MAIN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(DATASET_PATH)
    df = engineer_features(df)

    feature_cols = [c for c in df.columns if c != LABEL_COL]

    # Experiment 1 — Binary
    bin_results, rf_bin, gb_bin, scaler_bin, \
    X_test_bin, y_test_bin, rf_probs, gb_probs = binary_experiment(df, feature_cols)

    # Experiment 2 — Multi-class
    rf_mc, scaler_mc, X_test_mc, y_test_mc, \
    mc_preds, class_names, mc_auc = multiclass_experiment(df, feature_cols)

    # Adversarial test on binary RF
    base_r, adv_r, drop = adversarial_test(rf_bin, X_test_bin, y_test_bin)

    # Plots
    make_plots(y_test_bin, rf_probs, gb_probs,
               rf_mc, y_test_mc, mc_preds, class_names,
               feature_cols, rf_bin.feature_importances_)

    # Save models
    joblib.dump({"model": rf_bin, "scaler": scaler_bin,
                 "features": feature_cols, "mode": "binary"},
                "spoof_classifier_real.joblib")

    # Summary
    print("\n" + "=" * 55)
    print("FINAL SUMMARY")
    print("=" * 55)
    print(f"Dataset size:              510,530 samples")
    print(f"Binary RF ROC-AUC:         {bin_results['random_forest']['roc_auc']}")
    print(f"Binary GB ROC-AUC:         {bin_results['gradient_boosting']['roc_auc']}")
    print(f"Multi-class macro AUC:     {mc_auc:.4f}")
    print(f"Adversarial recall drop:   {drop:.4f}")
    print(f"Verdict: {'ROBUST' if drop<0.05 else 'MODERATE' if drop<0.15 else 'FRAGILE'}")
    print(json.dumps(bin_results, indent=2))