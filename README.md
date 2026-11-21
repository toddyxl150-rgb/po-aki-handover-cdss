# PO-AKI handover CDSS

Reference implementation for the manuscript **"A handover-window early-warning model for
postoperative acute kidney injury: a three-database development and external validation study"**
(submitted to *npj Digital Medicine*).

The system predicts **postoperative acute kidney injury (PO-AKI, KDIGO stage >= 1)** before it
becomes biochemically apparent, from routine intra-operative and ICU monitoring. It is built
around one methodological claim: the **12 hours spanning the end of surgery and the start of ICU
care (the handover window)** carry more usable signal than the first 12 hours after ICU admission.

| | Internal (INSPIRE, 10-fold) | External (eICU-CRD) |
|---|---:|---:|
| AUROC | 0.7463 ± 0.0392 | 0.5975 (95% CI 0.5696–0.6244) |
| AUPRC | 0.3896 | 0.1084 (95% CI 0.0931–0.1250) |
| Patients / windows | 9,543 / 60,000 | 2,437 / 93,429 |

Track 1 (ICU 0–12 h, `t_end = +12 h`) versus Track 2 (handover, `t_end = +8 h`) on the same
4,285 patients: AUROC **0.7898 → 0.8273** (Δ +0.0375, paired bootstrap P = 0.002), AUPRC
0.4743 → 0.5578, median lead time **5.2 h → 8.7 h**.

> **This repository contains source code only.** No patient record, no row-level extract, and no
> derived table with a patient identifier is distributed here. The three CSV files under
> `outputs/paper_assets/` are aggregate summaries (one row per model, per characteristic, or per
> statistical comparison). See [Compliance](#9-compliance-and-data-availability).

---

## 1. Contents

| File | Role | What it guarantees |
|---|---|---|
| `cdss_common.py` | shared constants | 10 channels, physiological plausibility ranges, fixed scaling bounds |
| `stats_utils.py` | statistics | local BH / Holm / Tukey implementations (no `statsmodels` dependency) |
| `01_cohort_builder.py` | cohort + labels | three database adapters, KDIGO staging at hourly resolution, attrition waterfall |
| `02_window_preprocessor.py` | windows | causal LOCF imputation, train-fold-only scaling, 3-state labelling, dual tracks |
| `03_feature_engineering.py` | features | 120 raw + 480 sub-window statistics + 30 Gini + 3 raw perfusion = **633** |
| `04_train_evaluate.py` | models | patient-grouped 10-fold CV, LightGBM / LR / MLP / KNN / MEWS-4 / GRU / RETAIN / TabNet, external validation with Platt calibration |
| `05_clinical_metrics.py` | alarm metrics | event recall, false-alarm rate, lead time, decision-curve analysis, EWMA + cooldown policies |
| `06_shap_analysis.py` | explanation | TreeSHAP on a patient-level hold-out |
| `07_track_compare.py` | track comparison | Track 1 vs Track 2 on identical patients and horizon |
| `08_paper_assets.py` | tables + figures | Tables A–D and Figures 2–4 at 300 DPI RGB |
| `09_verify_final_numbers.py` | provenance | re-derives every manuscript number from the results tree |
| `10_cohort_flowchart.py` | Figure 1 | three-column CONSORT diagram, numbers read from the attrition CSVs |
| `11_table1_baseline.py` | Table 1 | baseline characteristics across the three databases |
| `12_stats_npj.py` | statistics table | Dunn (Holm), 1,000× patient-level bootstrap CIs, DeLong test |
| `configs/default.yaml` | defaults | mirrors the CLI defaults, including `baseline_lookback_d: 180` |
| `demo/` | zero-data demo | synthetic eICU-shaped tables + one-click driver |
| `tools/check_no_phi.py` | release gate | three-gate scan for patient-level residue |

## 2. Architecture

```
 raw database tables                     hourly long table
 ┌────────────────┐    01     ┌──────────────────────────────┐
 │ INSPIRE v1.4.2 │──────────▶│ cohort_<db>.csv   (1 row/pt) │
 │ eICU-CRD v2.0  │           │ vitals_<db>.csv.gz (1 row/h) │
 │ VitalDB        │           └───────────────┬──────────────┘
 └────────────────┘                           │ 02  windows
                                ┌─────────────▼──────────────┐
                                │ windows_<track>_X.npy      │ (N, 10, 12) + obs mask
                                └─────────────┬──────────────┘
                                              │ 03  features
                                ┌─────────────▼──────────────┐
                                │ features_<track>.parquet   │ 633 columns
                                └─────────────┬──────────────┘
                                              │ 04  models
                     ┌────────────────────────▼─────────────────────────┐
                     │ 10-fold CV  ·  external validation  ·  SHAP      │
                     └────────────────────────┬─────────────────────────┘
                                              │ 05 07 08 11 12
                            tables, figures, statistics, audit report
```

Two design decisions are worth stating explicitly, because they are the parts most easily got
wrong in a sliding-window AKI pipeline:

* **Leakage control.** Imputation is last-observation-carried-forward only; the scaler is fitted
  on training folds only; a window is dropped when fewer than 50 % of its channel-hours were
  actually observed; and every window is truncated at `t_aki_onset` so that no post-onset
  physiology can enter a prediction.
* **Sampling-artefact control.** `t_end_h` — the distance from ICU admission to the end of the
  window — is a strong correlate of the outcome for purely administrative reasons. It is excluded
  from the feature matrix, and removing it was verified by ablation to *raise* AUPRC.

## 3. Installation

Python **3.13** (developed on 3.13.14), CPU only.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If the default PyPI wheel of PyTorch is too large, install the CPU build first:

```bash
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 4. Verify the code in about 90 seconds, without any data

No PhysioNet credential, no download, no configuration. The driver synthesises an eICU-shaped
directory, runs the real cohort builder on it, and then carries the result through windowing,
feature engineering, cross-validation and alarm metrics.

```bash
python demo/run_demo.py
```

Expected tail (timings vary with CPU; a 2020-class laptop takes 90–120 s):

```
==========================================================================
ALL CHECKS PASSED — 88 s total, 10 steps
==========================================================================
  stats_utils self-test         3.0s
  02 self-test                  5.4s
  03 self-test                  3.0s
  05 self-test                  6.9s
  synthetic eICU                3.8s
  01 cohort                     4.4s
  02 windows                   16.5s
  03 features                   3.8s
  04 train                     33.7s
  05 metrics                    7.2s
```

The four `--self-test` suites are not smoke tests. They assert the properties the manuscript
depends on:

| Suite | Assertions |
|---|---|
| `02 --self-test` | hourly bucketing, implausible-value rejection, **LOCF never carries a future value backwards**, training medians uncontaminated by test patients, no window survives past `t_aki_onset`, every positive label lies in `(t_end, t_end + 24]`, Track 2 `t_end ≡ +8 h`, `or_frac = 0` when no intra-operative data exist, fixed scaler maps bounds exactly to 0/1, tampering with `t >= 30 h` does not change the window ending at `t = 20 h` |
| `03 --self-test` | 633-column layout, Gini computed by the sorting identity matches the O(n²) definition, zero-variance branch returns 0.0 |
| `05 --self-test` | event recall, false-alarm rate, lead time and DCA on hand-computed cases |
| `stats_utils.py` | BH / Holm / Tukey against closed-form expectations |

To also exercise the Track 1 vs Track 2 code path (60 synthetic patients — a code-path check, not
a scientific result):

```bash
python demo/run_demo.py --with-track-compare
```

## 5. Data access

None of the three databases may be redistributed. Request them from the custodians:

| Database | Version | Access |
|---|---|---|
| **INSPIRE** | 1.4.2 | PhysioNet credentialed access — <https://physionet.org/content/inspire/1.4.2/> (requires CITI "Data or Specimens Only Research" training and a signed DUA) |
| **eICU-CRD** | 2.0 | PhysioNet credentialed access — <https://physionet.org/content/eicu-crd/2.0/> |
| **VitalDB** | public release | Registration at <https://vitaldb.net/> |

Expected layout (paths are passed on the command line or through environment variables; nothing
is hard-coded):

```
$PO_AKI_INSPIRE_ROOT/     operations.csv.gz  lab_data.csv  clinical_data.csv  ...
$PO_AKI_EICU_ROOT/        patient.csv.gz  lab.csv.gz  vitalPeriodic.csv.gz  ...
$PO_AKI_VITALDB_ROOT/     clinical_data.csv  *.vital
```

```bash
export PO_AKI_INSPIRE_ROOT=/path/to/inspire-1.4.2
export PO_AKI_EICU_ROOT=/path/to/eicu-crd-2.0
export PO_AKI_VITALDB_ROOT=/path/to/vitaldb
```

## 6. Full reproduction

Every command below is run from the repository root and writes into `outputs/`.

### 6.1 Cohorts and KDIGO labels

```bash
python 01_cohort_builder.py --db inspire  --out outputs
python 01_cohort_builder.py --db eicu     --limit 4000 --out outputs
python 01_cohort_builder.py --db vitaldb  --limit 3000 --out outputs
```

`--limit` is a documented run cap, not an exclusion: it appears as its own row in
`outputs/attrition_<db>.csv`. Drop it to process the full database.

### 6.2 Windows (both tracks)

```bash
python 02_window_preprocessor.py \
  --vitals outputs/vitals_inspire.csv.gz --cohort outputs/cohort_inspire.csv \
  --out outputs --track both --stage-target 1 --scaler fixed

python 02_window_preprocessor.py \
  --vitals outputs/vitals_eicu.csv.gz --cohort outputs/cohort_eicu.csv \
  --out outputs --track track1 --stage-target 1 --scaler fixed \
  --load-params outputs/preprocess_params.json
```

The second call reuses the source-database scaling parameters, so no target-database statistic
leaks into external validation. `--scaler fixed` maps every channel onto the same physiological
bounds in all three databases.

### 6.3 Features

```bash
python 03_feature_engineering.py \
  --npz outputs/windows_track1_X.npy --meta outputs/windows_track1_meta.csv \
  --out outputs --tag track1 --params outputs/preprocess_params.json

python 03_feature_engineering.py \
  --npz outputs/windows_track2_X.npy --meta outputs/windows_track2_meta.csv \
  --out outputs --tag track2
```

### 6.4 Training, external validation, explanation

```bash
# 10-fold internal CV (TabNet is split out because of runtime; see 6.5)
python 04_train_evaluate.py \
  --features outputs/features_track1.parquet --out outputs/cv_clean \
  --folds 10 --weights 10,25,50,100 --max-rows 60000 \
  --cohort-csv outputs/cohort_inspire.csv

# external validation on eICU-CRD (source-fitted models, 5 % Platt calibration set)
python 04_train_evaluate.py \
  --features outputs/features_track1.parquet --out outputs/cv_external_clean \
  --folds 10 --fast \
  --external-features outputs/features_eicu.parquet \
  --cohort-csv outputs/cohort_inspire.csv \
  --external-cohort outputs/cohort_eicu.csv --calib-frac 0.05

# SHAP on a patient-level hold-out
python 06_shap_analysis.py \
  --features outputs/features_inspire.parquet --out outputs/shap \
  --suffix _clean --cohort-csv outputs/cohort_inspire.csv
```

### 6.5 Track comparison, tables, figures, statistics

```bash
python 07_track_compare.py \
  --track1 outputs/features_inspire.parquet \
  --track2 outputs/features_inspire_track2.parquet \
  --out outputs/track_compare --folds 5

python 05_clinical_metrics.py --pred outputs/cv_clean/oof_LightGBM.csv \
  --threshold 0.3 --policy-multi

python 08_paper_assets.py --out outputs/paper_assets --fig-dir outputs/paper_figures
python 10_cohort_flowchart.py
python 11_table1_baseline.py
python 12_stats_npj.py
```

`12_stats_npj.py` runs 1,000 patient-level bootstrap replicates per metric and takes a few
minutes. TabNet, if you want it, is `04_train_evaluate.py --only-tabnet --max-rows 10000`.

## 7. Output layout

```
outputs/
├── cohort_<db>.csv              one row per patient (NOT distributed)
├── vitals_<db>.csv.gz           one row per hour x channel (NOT distributed)
├── attrition_<db>.csv           exclusion waterfall, one row per step
├── preprocess_params.json       medians and scaling bounds from the training folds
├── windows_<track>_X.npy        (N, 10, 12) float32
├── features_<track>.parquet     (N, 633)
├── cv_clean/                    10-fold metrics, OOF predictions
├── cv_external_clean/           external validation, cohorts A-D
├── shap/                        TreeSHAP importances
├── track_compare/               Track 1 vs Track 2
└── paper_assets/                tables, checklist, and the three shipped CSVs
```

Only the last directory is redistributed, and only three aggregate files from it.

## 8. Number provenance

Every number in the manuscript is derived, never transcribed. `09_verify_final_numbers.py`
re-reads the results tree, recomputes each quantity from its source CSV, and diffs it against the
manuscript at a tolerance of 5.1e-5 — tight enough to catch a value that was produced by
subtracting two already-rounded numbers.

```bash
python 09_verify_final_numbers.py \
  --manuscript manuscript_draft_en.md --strict
```

This script needs the complete `outputs/` tree (including patient-level intermediates), so it is
**not runnable inside this repository**; it exits with code 2 and lists the missing inputs rather
than failing with a traceback. It is shipped so that the provenance chain is inspectable.

Two definitions matter when re-deriving numbers:

* **AUPRC** is `sklearn.metrics.average_precision_score` everywhere (tables, statistics,
  manuscript). A hand-written cumulative-precision variant disagrees by ~7e-5 when prediction
  scores tie, so all code paths now use the sklearn estimator.
* **Recall@10 %** is computed over all AKI patients, not over positive windows only, because the
  clinical question is "what fraction of patients who will develop AKI are flagged".

## 9. Compliance and data availability

The release was built to satisfy the PhysioNet DUA and the *npj Digital Medicine* code
availability requirement simultaneously:

* **Excluded:** every `cohort_*.csv`, `vitals_*.csv(.gz)`, `oof_*.csv`, `*.parquet`, `*.npy`,
  any raw database path, and any backup file. `.gitignore` blocks these patterns so an accidental
  `git add -A` cannot reintroduce them.
* **Included:** source code, configuration, `requirements.txt`, `LICENSE`, this README, and three
  aggregate CSVs (`paper_tables_summary.csv`, `table1_baseline.csv`, `stat_tests_npj.csv`) whose
  row counts are 97, 13 and 25 respectively — no row corresponds to a patient.
* **Automated gate.** Before every push:

  ```bash
  python tools/check_no_phi.py
  ```

  Three independent gates must pass: (A) no row-level container by extension or filename pattern,
  (B) no identifier column and no 7+-digit record key in any shipped table, (C) no hard-coded
  host path or user name in any source or configuration file. The scanner exits non-zero on the
  first violation and is suitable for CI.

Patient-level data cannot be shared by the authors. Researchers who complete the PhysioNet
credentialing and DUA process can regenerate every intermediate file with the commands in
section 6.

## 10. License and citation

MIT — see `LICENSE`. The license covers code only and grants no rights to any patient data.

If you use this code, please cite the manuscript. A machine-readable `CITATION.cff` will be added
once the article receives a DOI.

Questions about the implementation: open an issue on this repository.
