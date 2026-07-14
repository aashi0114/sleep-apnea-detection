import pandas as pd


#########calculates per window apnea
df = pd.read_csv("dataset_csv/all_participants.csv")

# Compute apnea fraction per participant
participant_labels = (
    df.groupby("participant_id")["label_apnea"]
      .mean()
      .reset_index(name="apnea_fraction")
)

# Binary participant label (match paper logic: apnea vs no-apnea)
# Threshold can be adjusted; 0.1–0.2 is common
APNEA_FRAC_THRESHOLD = 0.15

participant_labels["participant_label"] = (
    participant_labels["apnea_fraction"] >= APNEA_FRAC_THRESHOLD
).astype(int)

participant_labels.head()



'''
#########Seperate CSV calculating overall apnea
#########Source: https://my.clevelandclinic.org/health/articles/apnea-hypopnea-index-ahi

# Load window-level data
df = pd.read_csv("dataset_csv/all_participants.csv")

# Aggregate to participant level
summary = (
    df.groupby("participant_id")
      .agg(
          n_windows=("label_apnea", "count"),
          n_apnea_windows=("label_apnea", "sum")
      )
      .reset_index()
)

# Estimate AHI (events/hour)
summary["ahi_estimated"] = (
    summary["n_apnea_windows"] / summary["n_windows"] * 60
)

# Binary apnea label (OSA vs normal)
summary["apnea_label"] = (summary["ahi_estimated"] >= 5).astype(int)

# Severity classification (optional but useful)
def ahi_to_severity(ahi):
    if ahi < 5:
        return "normal"
    elif ahi < 15:
        return "mild"
    elif ahi < 30:
        return "moderate"
    else:
        return "severe"

summary["severity"] = summary["ahi_estimated"].apply(ahi_to_severity)

# Save participant-level CSV
summary.to_csv("dataset_csv/overall_ahi_label.csv", index=False)

summary.head()
'''