import pandas as pd
import numpy as np

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

import time
import psutil
import os

process = psutil.Process(os.getpid())
start_time = time.time()
start_mem = process.memory_info().rss / (1024 ** 2)  # MB


RANDOM_STATE = 42

FEATURES_CSV = "dataset_csv/all_participants.csv"
SPLIT_CSV = "dataset_csv/participant_split.csv"
LABELS_CSV = "dataset_csv/overall_ahi_label.csv"  # participant-level apnea_label (AHI>=5)

# -----------------------------
# Helpers
# -----------------------------
def bin_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
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

feature_cols = ["rr_mean", "rr_std", "rr_rmssd", "spo2_mean", "spo2_min", "spo2_std"]

train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test"].copy()

X_train = train_df[feature_cols]
y_train = train_df["label_apnea"].astype(int)
groups  = train_df["participant_id"].astype(str).values

X_test  = test_df[feature_cols]
y_test  = test_df["label_apnea"].astype(int)

print("Train participants:", train_df["participant_id"].nunique(), "| Train windows:", len(train_df))
print("Test participants:", test_df["participant_id"].nunique(), "| Test windows:", len(test_df))


# -----------------------------
# IMPORTANT: GPC scaling warning
# -----------------------------
# GaussianProcessClassifier is O(n^3) in number of training samples.
# With ~25k windows this will be too slow.
#
# To keep it practical AND still paper-aligned, we train GPC on a
# *participant-aggregated* dataset:
#   one row per participant (mean of features across windows)
#
# This keeps "participant-wise split" intact and still uses ECG+SpO2 features.
# -----------------------------

def aggregate_to_participant(df_part: pd.DataFrame, feature_cols: list):
    agg = (
        df_part.groupby("participant_id")
               .agg({**{c: "mean" for c in feature_cols},
                     "label_apnea": "mean"})
               .reset_index()
    )
    # Convert window-label mean -> participant apnea proxy (binary)
    # This is NOT the same as apnea_label (AHI>=5) but is useful as a training target
    # for participant-level modeling.
    agg["y_participant"] = (agg["label_apnea"] >= 0.5).astype(int)
    return agg

train_part = aggregate_to_participant(train_df, feature_cols)
test_part  = aggregate_to_participant(test_df, feature_cols)

X_train_p = train_part[feature_cols]
y_train_p = train_part["y_participant"].astype(int)
groups_p  = train_part["participant_id"].astype(str).values

X_test_p  = test_part[feature_cols]
# For window-level metrics, we still evaluate on windows later.
# For participant-level metrics, we'll use labels_df (AHI>=5) same as XGB.
print("\nUsing participant-aggregated training for GPC:")
print("Train participants:", len(train_part), "| Test participants:", len(test_part))


# -----------------------------
# GPC model + tuning
# -----------------------------
# Kernel choices: (constant * RBF) + white noise
# WhiteKernel helps handle sensor noise + feature noise.
base_kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=np.ones(len(feature_cols)), length_scale_bounds=(1e-2, 1e2)) \
              + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

gpc = GaussianProcessClassifier(kernel=base_kernel, random_state=RANDOM_STATE, max_iter_predict=200)

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", gpc),
])

# Keep tuning light to avoid very long runtimes
param_grid = {
    # You can switch between slightly different kernels by tweaking noise bounds or removing WhiteKernel,
    # but GridSearch over kernels can be very slow. We'll just tune the optimizer restarts.
    "clf__n_restarts_optimizer": [0, 1, 3],
}

cv = GroupKFold(n_splits=5)

search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv.split(X_train_p, y_train_p, groups=groups_p),
    n_jobs=-1,
    verbose=1,
    refit=True
)

search.fit(X_train_p, y_train_p)
best_model = search.best_estimator_

print("\nBest CV AUC (participant-aggregated target):", round(search.best_score_, 4))
print("Best params:", search.best_params_)
print("Learned kernel:", best_model.named_steps["clf"].kernel_)


# -----------------------------
# Evaluate on held-out test (WINDOW-LEVEL)
# -----------------------------
# We trained on participant-level aggregates, but you asked for the same metrics as XGB.
# We'll still generate window-level probabilities by applying the *participant-level predictor*
# to each window row (same participant gets same prob). This keeps a fair participant comparison.

# Predict participant probabilities
test_part_probs = best_model.predict_proba(X_test_p)[:, 1]
test_part_scores = test_part[["participant_id"]].copy()
test_part_scores["prob_apnea"] = test_part_probs

# Map participant prob -> each window
test_df_scored = test_df.merge(test_part_scores, on="participant_id", how="left")

y_test_win = test_df_scored["label_apnea"].astype(int).to_numpy()
p_test_win = test_df_scored["prob_apnea"].to_numpy()

test_auc_win = roc_auc_score(y_test_win, p_test_win)
acc, sens, spec = bin_metrics(y_test_win, p_test_win, thr=0.5)

print("\nWINDOW-LEVEL TEST METRICS")
print("AUC:", round(test_auc_win, 4))
print("Accuracy:", round(acc, 4))
print("Sensitivity:", round(sens, 4))
print("Specificity:", round(spec, 4))


# -----------------------------
# Participant-level evaluation (paper-style)
#   Use mean_prob per participant vs apnea_label (AHI>=5)
# -----------------------------
subject_scores = test_part_scores.rename(columns={"prob_apnea": "mean_prob"}).copy()
subject_scores = subject_scores.merge(labels_df, on="participant_id", how="inner")

subject_auc = roc_auc_score(subject_scores["apnea_label"], subject_scores["mean_prob"])

print("\nPARTICIPANT-LEVEL TEST METRICS")
print("AUC:", round(subject_auc, 4))


# -----------------------------
# Save predictions
# -----------------------------
test_df_out = test_df_scored[["participant_id", "window_index", "label_apnea", "prob_apnea"]].copy()
test_df_out.to_csv("dataset_csv/gpc_test_window_predictions.csv", index=False)

subject_scores.to_csv("dataset_csv/gpc_test_subject_scores.csv", index=False)

print("\nSaved:")
print("- dataset_csv/gpc_test_window_predictions.csv")
print("- dataset_csv/gpc_test_subject_scores.csv")
end_time = time.time()
end_mem = process.memory_info().rss / (1024 ** 2)  # MB

print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
print(f"Memory usage change: {end_mem - start_mem:.2f} MB")
print(f"Final memory usage: {end_mem:.2f} MB")
