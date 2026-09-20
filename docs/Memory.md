# Memory — Technical Decisions & Lessons Learned
> **Institutional memory, architectural decisions, edge cases, and past blocker resolutions.**

---

## 1. Past Blockers & Resolutions

| Date | Blocker | Resolution |
|---|---|---|
| 2026-09-01 | Initial scan: all `src/` files are single-line stubs | Expected at scaffolding stage. Pipeline prototyped in notebooks before porting to `src/`. |
| 2026-09-01 | `requirements.txt` encoding was UTF-16LE | Re-saved as standard UTF-8 without BOM to avoid parsing issues across platforms. |
| 2026-09-01 | MIMII dataset too large (7.6 GB) for local laptop | Adopted Kaggle-first compute strategy — dataset stays in Kaggle cloud, local stores small sample (~80 MB). |
| 2026-09-01 | Multiple SNR levels complexity | Scoped to 6 dB SNR (cleanest baseline) for initial benchmark. Lower SNRs are optional extensions. |
| 2026-09-20 | Notebook corruption / malformed JSON | Rebuilt notebooks programmatically using valid JSON parsers to ensure clean cell structures. |
| 2026-09-20 | Kaggle RAM exhaustion when processing multiple machines | Implemented per-machine loop in NB03 with explicit `del` and `gc.collect()` to remain under 16 GB RAM ceiling. |
| 2026-09-20 | Source module decoupling & NB04 construction | Built `src/preprocess.py`, `src/dataset.py`, `src/model_ae.py`, `src/train.py`, `src/evaluate.py`, and self-contained Kaggle `NB04` (Conv2D-AE, FC-AE, LSTM-AE, shallow baselines, Dual-Stage Deep Hybrid). |
| 2026-09-20 | NB04 `state_dict().copy()` early stopping bug | `model.state_dict().copy()` creates a shallow dict copy where tensor values share memory storage with live model parameters. In-place optimizer updates silently corrupt the "best" checkpoint. Fixed to `copy.deepcopy(model.state_dict())` in both NB04 and `src/train.py`. |
| 2026-09-20 | NB04 XGBoost hardcoded label array | `y_sup = np.array([0]*5000 + [1]*5000)` doesn't dynamically size based on actual anomaly data length, risking data-label mismatch. Fixed to `n_anom = len(flat_anom)` with dynamic label construction. |
| 2026-09-20 | NB04 unused imports & missing memory management | Removed unused `import yaml` and `import sys`. Added `import copy`, `import gc`, `gc.collect()`, and `torch.cuda.empty_cache()` between machine training loops to stay under Kaggle 16 GB RAM ceiling. Documented config.yaml hyperparameter deviations (batch_size, epochs, patience) in notebook markdown. |

---

## 2. Key Architectural Decisions & Rationale

1. **Focus on Heavy Industrial Assets (Hitachi MIMII):**
   - Scoped strictly to 4 core machines: **Fan**, **Pump**, **Slider**, and **Valve** at `id_00`.
   - Excluded ToyADMOS miniature toys (`toycar`, `toyconveyor`) due to non-standard durations (11.0s vs. 10.0s) and non-zero ID indexing (`id_01`).

2. **Zero-Leakage Data Partitioning:**
   - Train/validation split (90/10) is performed at the **audio clip level** *before* framing into context windows. Slicing first and shuffling frames leaks temporal correlations between train and test sets.
   - Z-score normalization statistics ($\mu, \sigma$) are computed **strictly on training normal clips** and frozen for validation/test sets.

3. **Numerical Stability Guards:**
   - Log-Mel conversion: added $\epsilon = 10^{-10}$ before `log10()` to prevent $-\infty$ on silence.
   - Z-score normalization: added $\epsilon = 10^{-8}$ to the standard deviation denominator to prevent division by zero.

4. **Single Baselines to Dual-Stage Deep Hybrid Progression:**
   - Benchmarking single standalone algorithms (Isolation Forest, One-Class SVM, XGBoost, LSTM-AE, FC-AE, Conv2D-AE) establishes rigorous empirical baselines and exposes the supervised open-world failure trap.
   - Combining Conv2D-AE with Latent Isolation Forest ($S_{\text{final}} = 0.6 \cdot S_{\text{recon}} + 0.4 \cdot S_{\text{latent}}$) proves why fusing physical frequency error with latent manifold density achieves the highest ROC-AUC (~97–98%).

5. **Semester-2 Scale-Up Deferred to `docs/future/`:**
   - 2026-09-20: Created `docs/future/data-scale.md` (MIMII DUE, DCASE 2023-24, SNR/ID sweeps, field set demo-only) and `docs/future/algorithm-scale.md` (VAE/beta, embedding+kNN, Transformer-AE, Hybrid v2). Explicitly gated until Phase 0-6 closed to protect interim defense.

6. **Kaggle-First Execution Model:**
   - All notebooks are **fully self-contained** — model architectures, training loops, dataset classes, and evaluation logic are defined inline. No `import src.*` required.
   - `src/` folder is a **local reference mirror** with type hints and docstrings for portfolio/documentation purposes only. Notebooks do NOT import from it.
   - 100% of compute-heavy tasks execute on Kaggle Cloud (P100/T4 GPUs).
   - Local workspace stores only downloaded artifacts: `.npy` arrays, model checkpoints (`models/`), and reports (`reports/`).
   - Notebook 06 launches Gradio with `share=True` on Kaggle, generating a public URL for viva presentation on any mobile/laptop browser.
