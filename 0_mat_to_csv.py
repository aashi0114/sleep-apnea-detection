import os
import numpy as np
import pandas as pd
import scipy.io


DATASET_ROOT = "dataset_raw" 
RR_DIR = os.path.join(DATASET_ROOT, "RR")
SAT_DIR = os.path.join(DATASET_ROOT, "SAT")
LABEL_DIR = os.path.join(DATASET_ROOT, "LABELS")

OUTPUT_DIR = os.path.join(DATASET_ROOT, "dataset_csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def loadmat_vars(path: str) -> dict:
    """Load .mat and return only non-metadata variables."""
    mat = scipy.io.loadmat(path)
    return {k: v for k, v in mat.items() if not k.startswith("__")}

def pick_primary_array(var_dict: dict) -> np.ndarray:
    """
    For RR/SAT .mat files (unknown variable names), pick the 'most likely' payload.
    Heuristic: choose the ndarray with the largest number of elements.
    """
    candidates = []
    for k, v in var_dict.items():
        if isinstance(v, np.ndarray):
            candidates.append((k, v, v.size))
    if not candidates:
        raise ValueError("No ndarray variables found.")
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][1]

def pick_labels_array(var_dict: dict) -> np.ndarray:
    """
    For LABELS, prefer salida_man_1m if present, else largest array.
    """
    if "salida_man_1m" in var_dict:
        return var_dict["salida_man_1m"]
    if "salida_man" in var_dict:
        return var_dict["salida_man"]
    return pick_primary_array(var_dict)

def unwrap_to_1d_numeric(x) -> np.ndarray:
    """
    Convert MATLAB cell/array/number into a 1D float numpy array.
    Handles:
    - numeric arrays
    - object arrays (cell arrays)
    - nested object arrays
    - empty arrays
    """
    if x is None:
        return np.array([], dtype=float)

    # If it's a numpy array
    if isinstance(x, np.ndarray):
        # MATLAB cell arrays -> dtype object
        if x.dtype == object:
            parts = []
            for el in x.ravel():
                v = unwrap_to_1d_numeric(el)
                if v.size:
                    parts.append(v)
            return np.concatenate(parts).astype(float) if parts else np.array([], dtype=float)

        # Numeric ndarray
        return np.asarray(x, dtype=float).ravel()

    # Scalars (int/float)
    try:
        return np.array([float(x)], dtype=float)
    except Exception:
        return np.array([], dtype=float)

def as_window_list(arr: np.ndarray) -> list:
    """
    Represent RR/SAT as a Python list of per-window objects.
    - If arr is object array: each element is a window (cell).
    - Else: treat it as a vector of scalar-per-window values.
    """
    arr = np.asarray(arr).squeeze()

    # Common case: object array where each element is a window
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return list(arr.ravel())

    # Numeric arrays:
    # If it's 2D like (N,1) or (1,N), squeeze handles it. Now arr is 1D (N,)
    if arr.ndim == 1:
        return list(arr)

    # Fallback: flatten
    return list(arr.ravel())

# -----------------------
# Feature functions
# -----------------------
def rr_features(rr_vec: np.ndarray) -> dict:
    """
    rr_vec: sequence of RR intervals in ms (or similar)
    """
    rr_vec = rr_vec[np.isfinite(rr_vec)]
    if rr_vec.size < 2:
        return {"rr_mean": np.nan, "rr_std": np.nan, "rr_rmssd": np.nan}

    diff = np.diff(rr_vec)
    rmssd = np.sqrt(np.mean(diff * diff)) if diff.size else np.nan
    return {
        "rr_mean": float(np.mean(rr_vec)),
        "rr_std": float(np.std(rr_vec, ddof=0)),
        "rr_rmssd": float(rmssd),
    }

def spo2_features(spo2_vec: np.ndarray) -> dict:
    spo2_vec = spo2_vec[np.isfinite(spo2_vec)]
    if spo2_vec.size < 1:
        return {"spo2_mean": np.nan, "spo2_min": np.nan, "spo2_std": np.nan}
    return {
        "spo2_mean": float(np.mean(spo2_vec)),
        "spo2_min": float(np.min(spo2_vec)),
        "spo2_std": float(np.std(spo2_vec, ddof=0)),
    }

# -----------------------
# Main conversion
# -----------------------
rows = []
files = sorted([f for f in os.listdir(LABEL_DIR) if f.endswith(".mat")])

for fname in files:
    pid = fname.replace(".mat", "")

    rr_path = os.path.join(RR_DIR, fname)
    sat_path = os.path.join(SAT_DIR, fname)
    lab_path = os.path.join(LABEL_DIR, fname)

    if not (os.path.exists(rr_path) and os.path.exists(sat_path)):
        print(f"Skipping {pid}: missing RR or SAT file.")
        continue

    # Load mats
    rr_vars = loadmat_vars(rr_path)
    sat_vars = loadmat_vars(sat_path)
    lab_vars = loadmat_vars(lab_path)

    rr_arr = pick_primary_array(rr_vars)
    sat_arr = pick_primary_array(sat_vars)
    lab_arr = pick_labels_array(lab_vars)

    # Convert to per-window lists
    rr_windows = as_window_list(rr_arr)
    sat_windows = as_window_list(sat_arr)

    labels = np.asarray(lab_arr).squeeze().ravel()
    labels = labels.astype(int)

    # Align lengths (truncate to shared min)
    n = min(len(rr_windows), len(sat_windows), len(labels))
    if n == 0:
        print(f"Skipping {pid}: empty after loading.")
        continue

    # Build rows
    for i in range(n):
        rr_vec = unwrap_to_1d_numeric(rr_windows[i])
        sat_vec = unwrap_to_1d_numeric(sat_windows[i])

        feats = {}
        feats.update(rr_features(rr_vec))
        feats.update(spo2_features(sat_vec))

        rows.append({
            "participant_id": pid,
            "window_index": i,
            **feats,
            "label_apnea": int(labels[i]),
        })

    print(f"Processed {pid}: {n} windows")

df = pd.DataFrame(rows)

out_all = os.path.join(OUTPUT_DIR, "apnea_rr_spo2_features.csv")
df.to_csv(out_all, index=False)

print(f"\nSaved: {out_all}")
print(df.head())


'''

#for creating seperate CSV files
import os
import pandas as pd

# Path to your combined file
combined_csv_path = os.path.join(OUTPUT_DIR, "apnea_rr_spo2_features.csv")

# Load it (or skip this if you already have df in memory)
df = pd.read_csv(combined_csv_path)

# Folder to save per-participant files (same folder)
per_participant_dir = OUTPUT_DIR
os.makedirs(per_participant_dir, exist_ok=True)

for pid, g in df.groupby("participant_id"):
    out_path = os.path.join(per_participant_dir, f"{pid}.csv")
    g.to_csv(out_path, index=False)

print(f"Saved {df['participant_id'].nunique()} per-participant CSV files to: {per_participant_dir}")
'''