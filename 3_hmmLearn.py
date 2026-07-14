import pandas as pd
import numpy as np

from hmmlearn.hmm import GaussianHMM

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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

feature_cols = ["rr_mean", "rr_std", "rr_rmssd", "spo2_mean", "spo2_min", "spo2_std"]


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

def stable_sigmoid(x):
    # convert logit to probability stably
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def sequences_from_df(df_seq: pd.DataFrame, feature_cols: list):
    """
    Returns:
      X_concat: concatenated observations (N_total, D)
      lengths: list of sequence lengths per participant
      pids: list of participant IDs in same order as lengths
    """
    X_list = []
    lengths = []
    pids = []
    for pid, g in df_seq.groupby("participant_id"):
        g = g.sort_values("window_index")
        X = g[feature_cols].to_numpy(dtype=float)
        if len(X) == 0:
            continue
        X_list.append(X)
        lengths.append(len(X))
        pids.append(pid)
    if not X_list:
        return np.zeros((0, len(feature_cols))), [], []
    return np.vstack(X_list), lengths, pids


# -----------------------------
# Load + merge split
# -----------------------------
df = pd.read_csv(FEATURES_CSV)
split_df = pd.read_csv(SPLIT_CSV)[["participant_id", "split"]]
labels_df = pd.read_csv(LABELS_CSV)[["participant_id", "apnea_label"]]

df = df.merge(split_df, on="participant_id", how="inner")

train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test"].copy()

print("Train participants:", train_df["participant_id"].nunique(), "| Train windows:", len(train_df))
print("Test participants:", test_df["participant_id"].nunique(), "| Test windows:", len(test_df))


# -----------------------------
# Preprocess features (fit ONLY on training windows)
# -----------------------------
prep = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

prep.fit(train_df[feature_cols])

train_df[feature_cols] = prep.transform(train_df[feature_cols])
test_df[feature_cols]  = prep.transform(test_df[feature_cols])


# -----------------------------
# Train two HMMs: one for apnea windows, one for non-apnea windows
# -----------------------------
# IMPORTANT: We build sequences per participant, but we only keep windows of a given class
# inside each participant so we can fit "apnea-dynamics" vs "non-apnea-dynamics".
# This is a simple, effective approach and keeps code small.

def fit_hmm_for_class(df_train: pd.DataFrame, y_value: int, n_states: int = 2):
    df_c = df_train[df_train["label_apnea"].astype(int) == y_value].copy()

    # build sequences from remaining windows (still ordered by time)
    X_concat, lengths, _ = sequences_from_df(df_c, feature_cols)

    if len(lengths) == 0 or X_concat.shape[0] < (n_states * 10):
        raise ValueError(
            f"Not enough data to fit HMM for class {y_value}. "
            f"Need more windows; got {X_concat.shape[0]}."
        )

    hmm = GaussianHMM(
        n_components=n_states, #n_states=2 
        covariance_type="diag",
        n_iter=200,
        random_state=RANDOM_STATE, #random_state=42
        verbose=False
    )
    hmm.fit(X_concat, lengths)
    return hmm

# You can try 2 or 3 states; 2 keeps it interpretable (apnea-ish vs non-apnea-ish substate)
HMM_STATES = 2

hmm_nonapnea = fit_hmm_for_class(train_df, y_value=0, n_states=HMM_STATES)
hmm_apnea    = fit_hmm_for_class(train_df, y_value=1, n_states=HMM_STATES)

print("\nTrained HMMs:")
print("Non-apnea HMM states:", hmm_nonapnea.n_components)
print("Apnea HMM states:", hmm_apnea.n_components)


# -----------------------------
# Score test windows: log-likelihood ratio -> probability
# -----------------------------
# We score each participant sequence under each HMM, then distribute a per-time-step
# score using posteriors (simpler: score each window independently with emission likelihoods).
# hmmlearn does not expose emission probs directly per timestep in a clean API, so we do:
# - Use score_samples to get per-timestep log likelihood via "logprob" and posteriors, then
#   approximate per-timestep contribution with logsumexp on emissions is complex.
#
# Minimal, reliable approach:
# - Score each window independently by treating it as length-1 sequence under each model.
# This ignores transitions at test-time, but still uses the trained emission distributions,
# and keeps the method easy and stable.

def score_windows_independent(hmm_model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    # Each row scored as a length-1 sequence
    scores = np.zeros(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        scores[i] = hmm_model.score(X[i:i+1], lengths=[1])
    return scores

X_test_mat = test_df[feature_cols].to_numpy(dtype=float)
y_test = test_df["label_apnea"].astype(int).to_numpy()

ll_apnea = score_windows_independent(hmm_apnea, X_test_mat)
ll_non   = score_windows_independent(hmm_nonapnea, X_test_mat)

# Log-likelihood ratio as a logit
logit = ll_apnea - ll_non
prob_apnea = stable_sigmoid(logit)

# -----------------------------
# WINDOW-LEVEL METRICS
# -----------------------------
test_auc = roc_auc_score(y_test, prob_apnea)
acc, sens, spec = bin_metrics(y_test, prob_apnea, thr=0.5)

print("\nWINDOW-LEVEL TEST METRICS")
print("AUC:", round(test_auc, 4))
print("Accuracy:", round(acc, 4))
print("Sensitivity:", round(sens, 4))
print("Specificity:", round(spec, 4))


# -----------------------------
# PARTICIPANT-LEVEL METRICS (paper-style)
# Aggregate window probs -> mean_prob, compare vs apnea_label (AHI>=5)
# -----------------------------
tmp = test_df[["participant_id", "window_index"]].copy()
tmp["prob_apnea"] = prob_apnea

subject_scores = (
    tmp.groupby("participant_id")
       .agg(mean_prob=("prob_apnea", "mean"))
       .reset_index()
)

subject_scores = subject_scores.merge(labels_df, on="participant_id", how="inner")
subject_auc = roc_auc_score(subject_scores["apnea_label"], subject_scores["mean_prob"])

print("\nPARTICIPANT-LEVEL TEST METRICS")
print("AUC:", round(subject_auc, 4))


# -----------------------------
# Save predictions
# -----------------------------
out_win = test_df[["participant_id", "window_index", "label_apnea"]].copy()
out_win["prob_apnea"] = prob_apnea
out_win.to_csv("dataset_csv/hmm_test_window_predictions.csv", index=False)

subject_scores.to_csv("dataset_csv/hmm_test_subject_scores.csv", index=False)

print("\nSaved:")
print("- dataset_csv/hmm_test_window_predictions.csv")
print("- dataset_csv/hmm_test_subject_scores.csv")
end_time = time.time()
end_mem = process.memory_info().rss / (1024 ** 2)  # MB

print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
print(f"Memory usage change: {end_mem - start_mem:.2f} MB")
print(f"Final memory usage: {end_mem:.2f} MB")