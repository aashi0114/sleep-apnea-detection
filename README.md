# Low-Cost Sleep Apnea Detection from ECG and SpO₂

Code for **"A Low-Cost, Interpretable Machine-Learning System for Sleep Apnea Detection Using ECG and SpO₂ Signals."**

Sleep apnea affects close to one billion people worldwide, and an estimated 80 to 90 percent of cases go undiagnosed. The diagnostic gold standard, in-laboratory polysomnography, uses more than 25 sensors and can cost thousands of dollars. Existing at-home screeners rely on a simple oxygen-desaturation threshold, which misses most apneic events, particularly in patients who do not consistently desaturate.

This project compares five interpretable machine-learning models against that threshold baseline, using six physiologically grounded features extracted from ECG and pulse oximetry. The best model reaches a participant-level AUC of **0.958**, against **0.694** for the threshold rule. A working hardware prototype integrating both sensors was built for approximately **twenty dollars**.

Preprint: [Zenodo DOI — add link]

---

## Results

Median of three runs, participant-level stratified 80/20 split.

| Model | Window AUC | Window Acc. | Window Sens. | Window Spec. | Window F1 | Participant AUC |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.916** | **0.878** | 0.657 | **0.932** | 0.74 | **0.958** |
| SVM-RBF | 0.909 | 0.880 | 0.598 | 0.948 | 0.71 | **0.958** |
| HMM | 0.892 | 0.758 | **0.910** | 0.721 | **0.78** | 0.944 |
| Gaussian process | 0.845 | 0.855 | 0.483 | 0.946 | 0.62 | 0.903 |
| Naive Bayes | 0.788 | 0.789 | 0.599 | 0.835 | 0.66 | 0.889 |
| Threshold baseline | 0.599 | 0.804 | 0.037 | 0.991 | 0.55 | 0.694 |

The threshold baseline reaches a sensitivity of only **0.037**. A simple desaturation rule correctly flags almost none of the true apneic minutes, which is exactly the failure mode that non-desaturating patients expose.

Trained models also generalized to the independent PhysioNet Apnea-ECG cohort. Because that dataset contains ECG only, external validation used the three ECG-derived features alone and still reached an **AUC of 0.86**.

---

## Features

Six features are computed per non-overlapping one-minute window.

**From the ECG (heart-rate variability):**
- `rr_mean` — mean R-R interval
- `rr_std` — standard deviation of R-R intervals
- `rr_rmssd` — root mean square of successive differences

**From pulse oximetry:**
- `spo2_mean`, `spo2_min`, `spo2_std`

Interpretable feature-based models were chosen over neural networks deliberately: for transparency (clinical trust), robustness on small imbalanced biomedical datasets, and low computational cost for real-time deployment on cheap hardware.

---

## Running the pipeline

Scripts run in numeric order.

```bash
pip install -r requirements.txt

python 0_mat_to_csv.py            # convert raw .mat recordings to per-window feature CSVs
python 1_classify_participants.py # assign participant-level apnea labels from window labels
python 2_8020_stratSample.py      # participant-level stratified 80/20 split (prevents subject leakage)

python 3_baseline.py              # 3% desaturation threshold baseline (emulates home kits)
python 3_XGBoost.py
python 3_SVMRBF.py
python 3_hmmLearn.py
python 3_BayesNet.py
python 3_Gaussian.py

python 4_ROCcurve.py              # window-level and participant-level ROC curves
```

Splitting is done **at the participant level**, not the window level. Windows from the same person never appear in both train and test, which would otherwise inflate results badly.

---

## Data

The datasets are **not included in this repository**, because they are distributed under their own licences and access terms. See [`data/README.md`](data/README.md) for how to obtain them.

- **HuGCDN2014-OXI** (primary, 83 overnight recordings): https://doi.org/10.17632/cdxs63gdzc.1
- **PhysioNet Apnea-ECG** (external validation): https://doi.org/10.13026/C23W2R

Place the raw recordings in `dataset_raw/` and the pipeline will generate `dataset_csv/`.

---

## Hardware

The prototype integrates a pulse-oximetry sensor and a single-lead ECG sensor for roughly twenty dollars. The trained XGBoost model runs on the captured signals to produce a nightly apnea classification, in place of the 25-plus sensor array used in clinical polysomnography.

---

## Citing

If you use this code, please cite the preprint:

> Gupta, A. (2026). *A Low-Cost, Interpretable Machine-Learning System for Sleep Apnea Detection Using ECG and SpO₂ Signals.* Zenodo. [DOI]

---

## Licence

MIT. See [LICENSE](LICENSE).
