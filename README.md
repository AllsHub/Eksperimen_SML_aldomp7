# Automated Data Preprocessing Pipeline — Breast Cancer

A reproducible, automated **data preprocessing pipeline** for the Breast Cancer
Wisconsin dataset, wired to a **GitHub Actions** workflow so that cleaning runs on every
push. This is the data-engineering stage of a larger breast-cancer ML system.

## What it does

The pipeline (`preprocessing/automate_aldomp7.py`) takes the raw dataset and produces a
clean, model-ready CSV:

1. **Load** — raw breast-cancer data into a DataFrame
2. **Drop duplicates**
3. **Remove outliers** — IQR method (1.5 × IQR fence across all features)
4. **Drop missing values**
5. **Export** — clean dataset to `preprocessing/namadataset_preprocessing/breastcancer_preprocessing.csv`

## CI Automation

`.github/workflows/preprocessing_workflow.yml` runs the preprocessing script
automatically on GitHub Actions, ensuring the cleaned dataset is always reproducible from
the raw source — no manual steps.

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `GitHub Actions`

## Repository Structure

```
.
├── preprocessing/
│   ├── Eksperimen_aldomp7.ipynb            # Exploratory analysis
│   ├── automate_aldomp7.py                 # Automated preprocessing script
│   └── namadataset_preprocessing/
│       └── breastcancer_preprocessing.csv  # Cleaned output
├── .github/workflows/
│   └── preprocessing_workflow.yml          # CI automation
├── breastcancer_raw.csv                    # Raw dataset
└── requirements.txt
```

## Run Locally

```bash
pip install -r requirements.txt
python preprocessing/automate_aldomp7.py
```
