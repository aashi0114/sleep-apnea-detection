import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib import font_manager, rcParams

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams


''''####CODE FOR PARTICIPANT CURVE
# Try to find Roboto
roboto_fonts = [f for f in font_manager.findSystemFonts() if "Roboto" in f]

if roboto_fonts:
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Roboto"]
else:
    print("Roboto not found, using default font.")

# -----------------------------
# CONFIG
# -----------------------------
SUBJECT_FILES = {
    "XGBoost": "dataset_csv/XGBoost/xgb_test_subject_scores.csv",
    "SVM-RBF": "dataset_csv/SVMRBF/svm_rbf_test_subject_scores.csv",
    "HMM": "dataset_csv/HMMLearn/hmm_test_subject_scores.csv",
    "Bayes Net": "dataset_csv/BayesNet/gnb_test_subject_scores.csv",
    "Gaussian Process": "dataset_csv/Gaussian/gpc_test_subject_scores.csv",
}

LABEL_COL = "apnea_label"
PROB_COL = "mean_prob"

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7, 7))

for model, path in SUBJECT_FILES.items():
    df = pd.read_csv(path)
    fpr, tpr, _ = roc_curve(df[LABEL_COL], df[PROB_COL])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model}")

plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Participant-Level ROC Curves")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

'''



######CODE FOR WINDOW CURVE
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

WINDOW_FILES = {
    "XGBoost": "dataset_csv/XGBoost/xgb_test_window_predictions.csv",
    "SVM-RBF": "dataset_csv/SVMRBF/svm_rbf_test_window_predictions.csv",
    "HMM": "dataset_csv/HMMLearn/hmm_test_window_predictions.csv",
    "Bayes Net": "dataset_csv/BayesNet/gnb_test_window_predictions.csv",
    "Gaussian Process": "dataset_csv/Gaussian/gpc_test_window_predictions.csv",
}

plt.figure(figsize=(7, 7))

for model, path in WINDOW_FILES.items():
    df = pd.read_csv(path)
    fpr, tpr, _ = roc_curve(df["label_apnea"], df["prob_apnea"])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f"{model}")

plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Window-Level ROC Curves")
plt.legend(loc="lower right", frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

