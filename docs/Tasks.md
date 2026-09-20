# Tasks — Project Roadmap & Milestone Tracker
> **Phase-by-phase task breakdown, completion status, deliverables, and next actions.**

---

## Task Summary Dashboard

| Phase | Description | Deliverable | Status |
|---|---|---|:---:|
| **Phase 0** | Scaffolding & Configs | Virtual env, folder tree, `configs/config.yaml` | ✅ Completed |
| **Phase 1** | Data Acquisition & EDA | `01-data-acquisition-and-inspection.ipynb`, `02-audio-exploration-and-visualization.ipynb` | ✅ Completed |
| **Phase 2** | Audio Preprocessing Engine | `03-preprocessing-engine.ipynb`, `src/preprocess.py`, `src/dataset.py` | 🟡 In Progress |
| **Phase 3** | Baselines & Hybrid Training | `04-model-training-and-baselines.ipynb`, `src/model_ae.py`, `src/train.py` | ⬜ Pending |
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

### Phase 2 — Audio Preprocessing Engine (Current Session)
- [x] Build `notebooks/03-preprocessing-engine.ipynb` (22 cells: 12 markdown + 10 code).
- [x] Implement STFT (1024 / 512) $\rightarrow$ 128 Mel bins $\rightarrow$ dB scaling $\rightarrow$ 5-frame context framing `(128, 5)`.
- [x] Implement leak-free clip-level 90/10 train/validation split.
- [x] Implement Z-score normalization computed strictly on training normal data with $\epsilon = 10^{-8}$.
- [x] Optimize memory with per-machine processing loop and garbage collection (`gc.collect()`).
- [ ] Run NB03 on Kaggle GPU/CPU runtime and save preprocessed `.npy` arrays and `reports/norm_stats.json`.
- [ ] Port tested preprocessing functions into `src/preprocess.py`.
- [ ] Port PyTorch Dataset and DataLoader into `src/dataset.py`.

### Phase 3 — Standalone Single Algorithms & Dual-Stage Hybrid Model
- [ ] Implement standalone model architectures in `src/model_ae.py` (Conv2D-AE, FC-AE, LSTM-AE).
- [ ] Implement shallow baselines (Isolation Forest, One-Class SVM on handcrafted DSP features).
- [ ] Implement supervised baseline (XGBoost / LightGBM) to demonstrate open-world failure trap.
- [ ] Implement Dual-Stage Deep Hybrid Model (Conv2D-AE + Latent Isolation Forest with normalized score fusion $\alpha = 0.6$).
- [ ] Implement training loop with early stopping, validation tracking, and model checkpointing in `src/train.py`.
- [ ] Create `notebooks/04-model-training-and-baselines.ipynb` and execute training on Kaggle GPU.

### Phase 4 — Multi-Model Comparative Evaluation & SOTA Benchmarking
- [ ] Implement inference and anomaly scoring routines in `src/evaluate.py`.
- [ ] Calibrate decision threshold ($\theta = P_{95}$) on held-out normal validation clips.
- [ ] Generate cross-model comparison table (`reports/model_comparison_table.csv`) with ROC-AUC, pAUC (10% max FPR), Precision, Recall, F1, and Latency.
- [ ] Generate multi-model ROC curve comparison plot (`reports/multi_model_roc_curves.png`).
- [ ] Generate Explainable AI (XAI) difference spectrogram heatmaps (`reports/diff_heatmap_<machine>.png`).
- [ ] Create `notebooks/05-evaluation-and-metrics.ipynb` for Kaggle execution.

### Phase 5 — Stage B Extension: Variational Autoencoder (VAE)
- [ ] Implement Variational Autoencoder (`src/model_vae.py`) with reparameterization trick ($z = \mu + \sigma \odot \epsilon$).
- [ ] Implement combined objective loss: $\mathcal{L}_{\text{MSE}} + \beta \cdot D_{\text{KL}}$.
- [ ] Benchmark domain shift robustness under machine speed / load variations.

### Phase 6 — Interactive Diagnostic Web Demo
- [ ] Create `notebooks/06-interactive-diagnostic-app.ipynb` with Gradio (`share=True`) or Streamlit + Cloudflare Tunnel for cloud live demo.
- [ ] Implement local offline Streamlit application (`app.py`).
- [ ] Features: Audio upload & player, model architecture selector, Mel-spectrogram viewer, reconstructed spectrogram, difference heatmap XAI, and real-time Health Diagnostic Gauge (🟢 Healthy / 🔴 Faulty).

---

## Immediate Next Actions

1. Port tested preprocessing logic from `notebooks/03-preprocessing-engine.ipynb` into `src/preprocess.py` and `src/dataset.py`.
2. Implement model architectures (Conv2D-AE, FC-AE, LSTM-AE) in `src/model_ae.py`.
3. Create `notebooks/04-model-training-and-baselines.ipynb` for Phase 3 training on Kaggle GPU.
