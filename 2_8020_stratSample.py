import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold

# -----------------------------
# CONFIG
# -----------------------------
FEATURES_CSV = "dataset_csv/all_participants.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# AASM adult thresholds (events/hour)
AHI_THRESHOLDS = {
    "normal_max": 5,      # <5
    "mild_max": 15,       # 5–14
    "moderate_max": 30    # 15–29
    # severe: >=30
}

# -----------------------------
# Helpers
# -----------------------------
def ahi_to_severity(ahi: float) -> str:
    if ahi < AHI_THRESHOLDS["normal_max"]:
        return "normal"
    elif ahi < AHI_THRESHOLDS["mild_max"]:
        return "mild"
    elif ahi < AHI_THRESHOLDS["moderate_max"]:
        return "moderate"
    else:
        return "severe"

def get_cohort_from_pid(pid: str) -> str:
    """
    Optional: keep original dataset cohort label for reporting only.
    Does NOT affect splitting.
    """
    pid = str(pid)
    if pid.startswith("ND"):
        return "ND"
    if pid.startswith("D"):
        return "D"
    if pid.startswith("C"):
        return "C"
    return "UNK"

def make_participant_table(df_windows: pd.DataFrame) -> pd.DataFrame:
    # Each row is a 1-minute window; AHI approx = (apnea_minutes / total_minutes) * 60
    p = (
        df_windows.groupby("participant_id")
        .agg(
            n_windows=("label_apnea", "count"),
            n_apnea_windows=("label_apnea", "sum")
        )
        .reset_index()
    )
    p["ahi_estimated"] = (p["n_apnea_windows"] / p["n_windows"]) * 60.0
    p["severity"] = p["ahi_estimated"].apply(ahi_to_severity)

    # Binary OSA presence (paper-style): AHI >= 5
    p["apnea_label"] = (p["ahi_estimated"] >= 5).astype(int)

    # Keep cohort (C/D/ND) for later analysis only
    p["cohort"] = p["participant_id"].apply(get_cohort_from_pid)

    return p

def split_participants(
    ptab: pd.DataFrame,
    stratify_on: str = "apnea_label",  # "apnea_label" (recommended) or "severity"
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    if stratify_on not in ptab.columns:
        raise ValueError(f"stratify_on='{stratify_on}' not found. Choose from: {list(ptab.columns)}")

    # If stratifying on severity, make sure every class has enough members.
    strat = ptab[stratify_on]
    counts = strat.value_counts(dropna=False)
    too_small = counts[counts < 2]
    if len(too_small) > 0:
        raise ValueError(
            f"Cannot stratify on '{stratify_on}' because some classes have <2 participants:\n{too_small}"
        )

    train_ids, test_ids = train_test_split(
        ptab["participant_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )
    return set(train_ids), set(test_ids)

# -----------------------------
# Load window-level data
# -----------------------------
df = pd.read_csv(FEATURES_CSV)

# Sanity: required columns
required = {"participant_id", "window_index", "label_apnea"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in features CSV: {missing}")

# -----------------------------
# Build participant-level labels (AHI, severity)
# -----------------------------
ptab = make_participant_table(df)

# Save participant-level table (recommended)
ptab.to_csv("dataset_csv/overall_ahi_label.csv", index=False)

print("Participants:", ptab["participant_id"].nunique())
print(ptab["severity"].value_counts())
print(ptab["apnea_label"].value_counts())

# -----------------------------
# Split (paper-faithful)
#   - DO stratify by outcome (apnea_label) or severity
#   - DO NOT stratify by cohort (C/D/ND)
# -----------------------------
# Recommended: stratify_on="apnea_label"
train_pids, test_pids = split_participants(ptab, stratify_on="apnea_label")

ptab_split = ptab.copy()
ptab_split["split"] = np.where(
    ptab_split["participant_id"].isin(train_pids), "train",
    np.where(ptab_split["participant_id"].isin(test_pids), "test", "unassigned")
)

# Sanity checks
assert (ptab_split["split"] != "unassigned").all(), "Some participants not assigned to train/test."
print("\nSplit breakdown by apnea_label:")
print(ptab_split.groupby(["split", "apnea_label"]).size())

print("\nSplit breakdown by severity:")
print(ptab_split.groupby(["split", "severity"]).size())

print("\nSplit breakdown by cohort (C/D/ND):")
print(ptab_split.groupby(["split", "cohort"]).size())

# Save it
ptab_split.to_csv("dataset_csv/participant_split.csv", index=False)
print("\nSaved: dataset_csv/participant_split.csv")
