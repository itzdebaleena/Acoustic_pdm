# Tasks — Project Roadmap & Milestone Tracker
> **Phase-by-phase task breakdown, completion status, deliverables, and next actions.**

---

## Task Summary Dashboard

| Phase | Description | Deliverable | Status |
|---|---|---|:---:|
| **Phase 0** | Scaffolding & Configs | Virtual env, folder tree, `configs/config.yaml` | ✅ Completed |
| **Phase 1** | Data Acquisition & EDA | `01-data-acquisition-and-inspection.ipynb`, `02-audio-exploration-and-visualization.ipynb` | ✅ Completed |
| **Phase 2** | Audio Preprocessing Engine | `03-preprocessing-engine.ipynb`, `src/preprocess.py`, `src/dataset.py` | ✅ Completed |
| **Phase 3** | Baselines & Hybrid Training | `04-model-training-and-baselines.ipynb`, `src/model_ae.py`, `src/train.py` | 🟡 In Progress |
| **Phase 4** | Multi-Model Benchmarking | `05-evaluation-and-metrics.ipynb`, `src/evaluate.py`, comparison table & ROC plots | ⬜ Pending |
| **Phase 5** | Stage B Extension: VAE | `src/model_vae.py` (Domain Shift on DCASE) | ⬜ Pending |
| **Phase 6** | Interactive Web Application | `06-interactive-diagnostic-app.ipynb` (Gradio/Tunnel) & `app.py` (Local Streamlit) | ⬜ Pending |

---

## Detailed Task Breakdown

### Phase 0 — Environment & Project Scaffolding
- [x] Create Python virtual environment (`.venv/`) and pin dependencies in `requirements.txt`.
- [x] Establish modular folder structure (`src/`, `configs/`, `data/`, `models/`, `notebooks/`, `reports/`, `docs/`).
- [x] Configure `.gitignore` for data, checkpoints, and reports while preserving `.gitkeep`.
- [x] Populate `configs/config.yaml` with centralized parameters for audio, features, training, baselines, and hybrid fusion.

### Phase 1 — Data Acquisition, Inspection & Waveform Exploration
- [x] Build `notebooks/01-data-acquisition-and-inspection.ipynb` (dataset integrity audit, summary statistics, catalog generation `reports/indexed_dataset.csv`).
- [x] Build `notebooks/02-audio-exploration-and-visualization.ipynb` (waveform plots, log Mel-spectrograms, difference heatmaps for 4 industrial machines).
- [x] Scope dataset strictly to 4 Hitachi MIMII machines (**Fan**, **Pump**, **Slider**, **Valve** at `id_00`, 6 dB SNR; 5,101 clips total).
- [x] Verify zero JSON errors and clean execution across NB01 and NB02.

### Phase 2 — Audio Preprocessing Engine
- [x] Build `notebooks/03-preprocessing-engine.ipynb` (22 cells: 12 markdown + 10 code).
- [x] Implement STFT (1024 / 512) $\rightarrow$ 128 Mel bins $\rightarrow$ dB scaling $\rightarrow$ 5-frame context framing `(128, 5)`.
- [x] Implement leak-free clip-level 90/10 train/validation split.
- [x] Implement Z-score normalization computed strictly on training normal data with $\epsilon = 10^{-8}$.
- [x] Optimize memory with per-machine processing loop and garbage collection (`gc.collect()`).
- [x] Run NB03 on Kaggle GPU/CPU runtime and save preprocessed `.npy` arrays and `reports/norm_stats.json` (Verified: 1.13M blocks across 4 machines).
- [x] *(Local ref)* Mirror preprocessing functions in `src/preprocess.py` and `src/dataset.py`.

### Phase 3 — Standalone Single Algorithms & Dual-Stage Hybrid Model
- [x] Define Conv2D-AE, FC-AE, LSTM-AE architectures inline in NB04.
- [x] Define shallow baselines (Isolation Forest, One-Class SVM) inline in NB04.
- [x] Define supervised baseline (XGBoost) inline in NB04 to demonstrate open-world failure trap.
- [x] Define Dual-Stage Deep Hybrid Model (Conv2D-AE + Latent Isolation Forest with normalized score fusion $\alpha = 0.6$) inline in NB04.
- [x] Define training loop with early stopping, validation tracking, and `copy.deepcopy` checkpointing inline in NB04.
- [x] Create self-contained `notebooks/04-model-training-and-baselines.ipynb` (ready for Kaggle GPU execution).
- [x] *(Local ref)* Mirror architectures and training logic in `src/model_ae.py` and `src/train.py`.
- [ ] Run NB04 on Kaggle GPU and save trained model checkpoints (`models/*.pth`, `models/*.joblib`).

### Phase 4 — Multi-Model Comparative Evaluation & SOTA Benchmarking
- [x] *(Local ref)* Mirror scoring and metric routines in `src/evaluate.py`.
- [x] Create self-contained `notebooks/05-evaluation-and-metrics.ipynb` with all 7-model benchmarking, ROC curves, multi-machine evaluation, and XAI heatmaps (ready for Kaggle GPU execution).
- [ ] Run NB05 on Kaggle GPU (attach NB03 + NB04 outputs) and generate:
  - `reports/model_comparison_table.csv` (ROC-AUC, pAUC, Precision, Recall, F1)
  - `reports/multi_model_roc_curves.png`
  - `reports/multi_machine_hybrid_results.csv`
  - `reports/xai_difference_heatmaps.png`

### Phase 5 — Stage B Extension: Variational Autoencoder (VAE)
- [ ] Implement Variational Autoencoder (`src/model_vae.py`) with reparameterization trick ($z = \mu + \sigma \odot \epsilon$).
- [ ] Implement combined objective loss: $\mathcal{L}_{\text{MSE}} + \beta \cdot D_{\text{KL}}$.
- [ ] Benchmark domain shift robustness under machine speed / load variations.

### Phase 6 — Interactive Diagnostic Web Demo
- [ ] Create `notebooks/06-interactive-diagnostic-app.ipynb` with Gradio (`share=True`) for cloud live demo on Kaggle.
- [ ] Features: Audio upload & player, model architecture selector, Mel-spectrogram viewer, reconstructed spectrogram, difference heatmap XAI, and real-time Health Diagnostic Gauge (🟢 Healthy / 🔴 Faulty).

---

## Immediate Next Actions

1. Run `notebooks/04-model-training-and-baselines.ipynb` on Kaggle GPU (attach NB03 output as input dataset).
2. Download trained model checkpoints (`models/*.pth`, `models/*.joblib`) to local workspace.
3. Build `notebooks/05-evaluation-and-metrics.ipynb` for Phase 4 comparative benchmarking.
