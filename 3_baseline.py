import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import time
import psutil
import os

process = psutil.Process(os.getpid())
start_time = time.time()
start_mem = process.memory_info().rss / (1024 ** 2)  # MB

# -----------------------------
# CONFIG
# -----------------------------
FEATURES_CSV = "dataset_csv/all_participants.csv"
SPLIT_CSV = "dataset_csv/participant_split.csv"
LABELS_CSV = "dataset_csv/overall_ahi_label.csv"

DROP_PCT = 3.0                 # desaturation threshold (3% is common; try 4% too)
ROLL_BASELINE_MINUTES = 5      # rolling baseline window in minutes
PROB_SCALE = 6.0               # scales drop magnitude into [0,1]

# -----------------------------
# Helpers
# -----------------------------
def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)

def metrics_from_pred(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return acc, sens, spec

# -----------------------------
# Load + merge split
# -----------------------------
df = pd.read_csv(FEATURES_CSV)
split_df = pd.read_csv(SPLIT_CSV)[["participant_id", "split"]]
labels_df = pd.read_csv(LABELS_CSV)[["participant_id", "apnea_label"]]

df = df.merge(split_df, on="participant_id", how="inner")
test_df = df[df["split"] == "test"].copy()

# Sort and RESET index so group indices match .loc/.iloc
test_df = test_df.sort_values(["participant_id", "window_index"]).reset_index(drop=True)

print("Test participants:", test_df["participant_id"].nunique(), "| Test windows:", len(test_df))

# -----------------------------
# ODI-like baseline using relative drops from rolling baseline
# -----------------------------
# Use spo2_mean for ODI-style "drop" logic
test_df["spo2_mean_clean"] = pd.to_numeric(test_df["spo2_mean"], errors="coerce")

baseline_arr = np.empty(len(test_df), dtype=float)
filled_arr = np.empty(len(test_df), dtype=float)

# Process each participant sequence
for pid, g in test_df.groupby("participant_id", sort=False):
    idx = g.index  # now safe because we reset_index(drop=True)
    s = test_df.loc[idx, "spo2_mean_clean"].to_numpy(dtype=float)

    # Fill NaNs with participant median (or 95 as fallback)
    med = np.nanmedian(s)
    if not np.isfinite(med):
        med = 95.0
    s_filled = np.where(np.isfinite(s), s, med)

    # Rolling median baseline of *previous* minutes
    b = (
        pd.Series(s_filled)
          .rolling(ROLL_BASELINE_MINUTES, min_periods=1)
          .median()
          .shift(1)
          .fillna(method="bfill")
          .to_numpy(dtype=float)
    )

    filled_arr[idx] = s_filled
    baseline_arr[idx] = b

test_df["spo2_filled"] = filled_arr
test_df["baseline"] = baseline_arr

drop = test_df["baseline"].to_numpy() - test_df["spo2_filled"].to_numpy()

# Window-level prediction: desaturation event if drop >= DROP_PCT
pred_desat = (drop >= DROP_PCT).astype(int)

# Probability-like score for AUC: bigger drop => higher score
prob_apnea = np.clip(drop / PROB_SCALE, 0.0, 1.0)

# -----------------------------
# WINDOW-LEVEL METRICS
# -----------------------------
y_true_win = test_df["label_apnea"].astype(int).to_numpy()

window_auc = safe_auc(y_true_win, prob_apnea)
acc, sens, spec = metrics_from_pred(y_true_win, pred_desat)

print("\nWINDOW-LEVEL TEST METRICS (ODI-style SpO2 baseline)")
print("AUC:", round(window_auc, 4) if np.isfinite(window_auc) else window_auc)
print("Accuracy:", round(acc, 4))
print("Sensitivity:", round(sens, 4))
print("Specificity:", round(spec, 4))

# -----------------------------
# PARTICIPANT-LEVEL METRICS
# -----------------------------
# ODI_est events/hr ≈ (desat_minutes / total_minutes)*60
subject = (
    test_df.assign(desat=pred_desat, prob=prob_apnea)
           .groupby("participant_id")
           .agg(
               n_minutes=("desat", "count"),
               n_desat=("desat", "sum"),
               mean_prob=("prob", "mean")
           )
           .reset_index()
)
subject["odi_est"] = (subject["n_desat"] / subject["n_minutes"]) * 60.0

# Participant score for AUC: scale ODI into [0,1]
subject["score"] = np.clip(subject["odi_est"] / 30.0, 0.0, 1.0)

subject = subject.merge(labels_df, on="participant_id", how="inner")

participant_auc = safe_auc(subject["apnea_label"], subject["score"])

print("\nPARTICIPANT-LEVEL TEST METRICS (ODI-style SpO2 baseline)")
print("AUC:", round(participant_auc, 4) if np.isfinite(participant_auc) else participant_auc)

# -----------------------------
# Save outputs
# -----------------------------
out_win = test_df[["participant_id", "window_index", "label_apnea"]].copy()
out_win["spo2_mean"] = test_df["spo2_filled"]
out_win["baseline"] = test_df["baseline"]
out_win["drop"] = drop
out_win["prob_apnea"] = prob_apnea
out_win["pred_desat"] = pred_desat
out_win.to_csv("dataset_csv/spo2_odi_baseline_test_window_predictions.csv", index=False)

out_subj = subject[["participant_id", "apnea_label", "odi_est", "score", "mean_prob", "n_desat", "n_minutes"]].copy()
out_subj.to_csv("dataset_csv/spo2_odi_baseline_test_subject_scores.csv", index=False)

print("\nSaved:")
print("- dataset_csv/spo2_odi_baseline_test_window_predictions.csv")
print("- dataset_csv/spo2_odi_baseline_test_subject_scores.csv")
end_time = time.time()
end_mem = process.memory_info().rss / (1024 ** 2)  # MB

print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
print(f"Memory usage change: {end_mem - start_mem:.2f} MB")
print(f"Final memory usage: {end_mem:.2f} MB")