# Future — Data Scale (Post Phase 0-6)

> **Status: NOT STARTED. Touch only after Phase 0-6 closed (NB04/NB05 run, `models/` + `reports/` populated, NB06 demo working).**
> This doc scopes the 8-month Semester-2 data expansion. No code changes required now.

---

## 1. Baseline Freeze (Do Not Reopen)

* Dataset: Kaggle `daisukelab/dc2020task2` (~7.6 GB), scoped to `fan/pump/slider/valve`, `id_00`, `6_dB` SNR, 5,101 clips — see `docs/Project.md`.
* Pipeline: STFT `1024/512` -> 128 Mel -> dB (`eps=1e-10`) -> 5-frame blocks `(128,5)` -> clip-level 90/10 split -> Z-score from train-normal only (`eps=1e-8`) — see `docs/Architecture.md`, `configs/config.yaml:18-28`.
* Current `configs/config.yaml:34-41`: `machine_types: [fan,pump,slider,valve]`, `machine_ids: [id_00]`, `snr_levels: [6_dB]`.

Semester-1 must finish with these frozen. All items below are additive.

---

## 2. Expansion Targets

### 2.1 MIMII DUE (Primary, DCASE 2021 Task 2 successor)
* Same 4 Hitachi machines, adds explicit `source/target` domains with speed/load/viscosity shift.
* Use case: true domain-shift benchmark. Train on source-normal, adapt with 10-shot target-normal, test on target-normal + target-anomaly.
* Config change: add `domain: [source, target]`, `shots: 10`. Keep `n_fft/hop/n_mels/context` unchanged.

### 2.2 DCASE 2023-2024 Task 2 (Variety)
* Adds `bandsaw, grinder, shaker, ToyTrain, ToyCar` + continuously varying operating conditions.
* Use case: "variety of machines" requirement. Tests cross-machine generalization, not just cross-condition.
* Note: ToyCar/ToyTrain durations and IDs differ — reuse NB01 catalog logic (`reports/indexed_dataset.csv`) to audit before NB03.

### 2.3 In-Dataset Sweeps (Zero-Download, Do First)
* SNR sweep: train `6_dB` -> test `6_dB / 0_dB / -6_dB`. Already on Kaggle input, no new download.
* ID sweep: train `id_00` -> test `id_01 / id_02` (where available).
* Metric: ROC-AUC drop `source->target`. Expected 5-15 pts drop; this table alone justifies Semester-2.

### 2.4 Self-Collected Field Set (Demo Only, Never Retrain Benchmark)
* 50-100 phone recordings (16 kHz mono, 10 s): bike idle vs misfire, car belt squeal, workshop grinder.
* Stored under `data/raw/field/` (gitignored). Run through frozen Phase-4 model in `app.py` as qualitative overlay.
* Reason: no public labeled car/bike factory dataset exists at MIMII quality. Field set proves transfer, does not replace benchmark.

---

## 3. Pipeline Impact

* NB01: extend catalog to `machine, id, snr, domain, split` columns. Output `reports/indexed_dataset_v2.csv` (keep v1 intact).
* NB03: loop `machine x domain x snr` with `del + gc.collect()` (existing RAM guard). Output `data/processed/*_{domain}_{snr}.npy` + `reports/norm_stats_{machine}_{domain}.json`.
* `mu/sigma` rule unchanged: compute strictly on `train-normal-source` per machine-domain. Freeze for all target/test.
* Kaggle DAG: NB03 output grows ~3-4x. Attach as versioned dataset to NB04/NB05. No change to `src/preprocess.py` / `src/dataset.py` signatures — only config values.

---

## 4. Evaluation Protocol (Semester-2)

* Per `(machine, target-domain)` report: ROC-AUC, pAUC (FPR<=0.1), F1 @ `theta=P95` calibrated on source-validation only.
* Tables: `reports/domain_shift_matrix.csv` (rows=train-domain, cols=test-domain), `reports/snr_robustness.csv`.
* Success bar: <8 pts ROC drop source->target after adaptation vs >15 pts without. If not met, algorithm scale (companion doc) is triggered.

---

## 5. Entry Checklist (After Phase 6)

1. `models/*.pth` + `reports/model_comparison_table.csv` exist from Phase 4.
2. Copy `configs/config.yaml` -> `configs/config_v2.yaml` for DUE/DCASE IDs. Never edit v1 in place.
3. Run SNR sweep first (1 Kaggle session). Only then download DUE (~2-5 GB subset).
