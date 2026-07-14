import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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
LABELS_CSV = "dataset_csv/overall_ahi_label.csv"  # for participant-level ground truth


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
labels_df = pd.read_csv(LABELS_CSV)[["participant_id", "apnea_label"]]  # participant-level OSA label

df = df.merge(split_df, on="participant_id", how="inner")

feature_cols = ["rr_mean", "rr_std", "rr_rmssd", "spo2_mean", "spo2_min", "spo2_std"]

train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test"].copy()

X_train = train_df[feature_cols]
y_train = train_df["label_apnea"].astype(int)
groups  = train_df["participant_id"].astype(str).values

X_test = test_df[feature_cols]
y_test = test_df["label_apnea"].astype(int)

print("Train participants:", train_df["participant_id"].nunique(), "| Train windows:", len(train_df))
print("Test participants:", test_df["participant_id"].nunique(), "| Test windows:", len(test_df))


# -----------------------------
# Model + tuning (participant-wise CV)
# -----------------------------
# Pipeline: impute missing -> XGBoost
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",     # fast CPU training
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", xgb),
])

# Small randomized search for minimal setup
param_dist = {
    "clf__n_estimators": [200, 400, 600],
    "clf__max_depth": [2, 3, 4, 5],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.7, 0.85, 1.0],
    "clf__min_child_weight": [1, 5, 10],
    "clf__reg_lambda": [1.0, 5.0, 10.0],
}

cv = GroupKFold(n_splits=5)

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=20,                     # keep small; increase if you want
    scoring="roc_auc",
    cv=cv.split(X_train, y_train, groups=groups),
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True,
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

print("\nBest CV AUC:", search.best_score_)
print("Best params:", search.best_params_)


# -----------------------------
# Evaluate on held-out test (window-level)
# -----------------------------
test_prob = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_prob)
acc, sens, spec = bin_metrics(y_test.to_numpy(), test_prob, thr=0.5)

print("\nWINDOW-LEVEL TEST METRICS")
print("AUC:", round(test_auc, 4))
print("Accuracy:", round(acc, 4))
print("Sensitivity:", round(sens, 4))
print("Specificity:", round(spec, 4))


# -----------------------------
# Participant-level evaluation (paper-style)
#   Aggregate window probs -> subject score
# -----------------------------
tmp = test_df[["participant_id"]].copy()
tmp["prob"] = test_prob

subject_scores = (
    tmp.groupby("participant_id")
       .agg(mean_prob=("prob", "mean"))
       .reset_index()
)

# Join participant-level ground truth apnea_label (AHI>=5)
subject_scores = subject_scores.merge(labels_df, on="participant_id", how="inner")

subject_auc = roc_auc_score(subject_scores["apnea_label"], subject_scores["mean_prob"])

print("\nPARTICIPANT-LEVEL TEST METRICS")
print("AUC:", round(subject_auc, 4))


# -----------------------------
# Save predictions (optional)
# -----------------------------
out_win = test_df[["participant_id", "window_index", "label_apnea"]].copy()
out_win["prob_apnea"] = test_prob
out_win.to_csv("dataset_csv/xgb_test_window_predictions.csv", index=False)

subject_scores.to_csv("dataset_csv/xgb_test_subject_scores.csv", index=False)

print("\nSaved:")
print("- dataset_csv/xgb_test_window_predictions.csv")
print("- dataset_csv/xgb_test_subject_scores.csv")
end_time = time.time()
end_mem = process.memory_info().rss / (1024 ** 2)  # MB

print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
print(f"Memory usage change: {end_mem - start_mem:.2f} MB")
print(f"Final memory usage: {end_mem:.2f} MB")
