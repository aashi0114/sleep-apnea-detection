# Data

The datasets used in this project are **not redistributed here**. Both are publicly
available under their own licences and access terms, and should be downloaded from
their original sources.

## Primary dataset: HuGCDN2014-OXI

83 continuous overnight recordings from Dr. Negrin University Hospital
(Las Palmas de Gran Canaria, Spain). ECG at 200 Hz, SpO2 at 50 Hz. Every one-minute
epoch was labelled apnea or non-apnea by an expert from simultaneous polysomnography,
following AASM guidelines.

The cohort has three groups:

- 38 healthy controls (AHI < 5)
- 34 obstructive sleep apnea patients (AHI 30 to 106.3) who desaturate during apneic episodes
- 11 obstructive sleep apnea patients (AHI 26.2 to 87.5) who do **not** consistently desaturate

That last group is the reason this dataset was chosen. Non-desaturating patients are
precisely the people an oxygen-threshold screener is most likely to miss, and few public
datasets include them.

**Download:** https://doi.org/10.17632/cdxs63gdzc.1 (Mendeley Data, CC BY 4.0)

## External validation: PhysioNet Apnea-ECG

Used to test generalization to an independent cohort. This dataset contains **ECG only**
and no oximetry, so external validation used the three ECG-derived features alone.

**Download:** https://doi.org/10.13026/C23W2R

## Expected layout

After downloading, arrange the raw files like this:

```
dataset_raw/
  RR/       # .mat files, one per participant
  SAT/      # .mat files, one per participant
  LABELS/   # .mat files, one per participant
```

Then run `0_mat_to_csv.py`, which will generate `dataset_csv/`.
