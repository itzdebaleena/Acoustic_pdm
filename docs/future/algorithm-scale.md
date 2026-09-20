# Future — Algorithm Scale (Post Phase 0-6)

> **Status: NOT STARTED. Touch only after Phase 0-6 closed and `docs/future/data-scale.md` SNR sweep is done.**
> This doc scopes the 8-month Semester-2 model upgrades. Baseline stays frozen as comparison anchor.

---

## 1. Baseline Freeze (Anchor)

From `docs/Project.md`, `docs/Architecture.md`, `configs/config.yaml:72-100`:

* Shallow: Isolation Forest (`n_estimators=100`), One-Class SVM (`nu=0.05`).
* Supervised trap: XGBoost (`max_depth=6`) — expected ~0.84 known / ~0.50 unseen.
* Deep: LSTM-AE (`hidden=64,L=2`), FC-AE (`[512,256,128]`), Conv2D-AE (`[1,32,64,128]`, `latent_dim=32`).
* Hybrid SOTA: `S_final = 0.6*S_recon + 0.4*S_latent` on `(1,128,5)` blocks, clip aggregation `A(y)=mean(s_n)`, `theta=P95`.

All 7 stay in `evaluation.benchmark_models` (`configs/config.yaml:112-119`). Never remove; new models are appended.

---

## 2. Upgrade Track (Priority Order)

### 2.1 VAE Family (Weeks 1-6) — Completes Phase 5 properly
* File: `src/model_vae.py` (currently stub). Implement Conv encoder -> `mu, logvar` -> `z = mu + sigma*eps` -> ConvTranspose decoder.
* Loss: `L = L_MSE + beta * D_KL`, sweep `beta in [0.1, 0.5, 1.0, 4.0]` (`configs/config.yaml:103-106`).
* CVAE follow-up: condition on `machine_id/domain` embedding to factor out known shifts.
* Score: use `L_MSE` only for anomaly (standard), report KL separately for analysis.
* Why: continuous regularized latent -> smaller source->target drop than plain AE.

### 2.2 Pretrained Embedding + kNN (Weeks 7-14) — Highest ROI for Domain Shift
* Freeze PANNs / BEATs / YAMNet as Mel-block feature extractor, fit kNN or single-Gaussian on source-normal embeddings.
* No large GPU training. Input adapter only: resample/repeat `(128,5)` blocks to backbone expected window; reuse frozen `mu/sigma` from data-scale.
* Why: transfer features are empirically most robust to SNR/speed shift in DCASE 2021-2023. Expected to beat Conv2D-AE on target domains.

### 2.3 Transformer-AE (Weeks 15-22) — Sequential SOTA
* Input: sequence of 5 frame vectors (128-dim each) + positional encoding, 2-4 layer Transformer encoder/decoder, `latent_dim=32` for comparability.
* Training: same `training:` block (`configs/config.yaml:54-61` — `lr=1e-3, epochs=50, patience=10, seed=42`).
* Why: explicit temporal attention vs Conv local receptive field. Compare head-to-head on slider/valve (non-stationary faults).

### 2.4 Flows / GAN (Optional, Weeks 23-28, Pick One)
* Normalizing Flow (MAF/RealNVP) on 32-d AE latents for exact likelihood scoring, OR AnoGAN-style discriminator score.
* Gate: only if 2.1-2.3 still show >10 pts domain drop. Higher compute, lower stability — do not start here.

### 2.5 Hybrid v2 (Throughout)
* Current: `alpha_recon=0.6` fixed (`configs/config.yaml:99-100`). Upgrade to per-machine learned `alpha` on source-validation, plus `max` aggregation option for impulsive faults (valve clicks).
* Formula unchanged: `S_final = alpha*S_recon_tilde + (1-alpha)*S_latent_tilde`. Log chosen `alpha` per machine in reports.

---

## 3. Training & Scoring Rules (Unchanged)

* Seed 42 everywhere. Clip-level split before framing. `copy.deepcopy(state_dict)` for checkpoints (see `docs/Memory.md` NB04 bug).
* Threshold: `theta=P95` on source-normal validation only. Never recalibrate on target-anomaly. Report Gaussian `mu+3sigma` as secondary.
* Block->clip: `mean` default, `max` as ablation column.

---

## 4. Evaluation Additions

* Append to `benchmark_models`: `vae_beta, cvae, embedding_knn, transformer_ae, [flow|gan], hybrid_v2`.
* Outputs: `reports/model_comparison_table_v2.csv`, `reports/domain_shift_matrix_v2.csv`, `reports/multi_model_roc_curves_v2.png`.
* Ablations required for thesis: `beta` sweep curve, `alpha` sweep curve, `mean vs max` aggregation, `embedding frozen vs fine-tuned`.
* Success bar: Hybrid v2 or embedding-kNN within 5 pts of source ROC on target domains, and best mean-target ROC > Phase-4 best source ROC minus 8 pts.

---

## 5. Entry Checklist (After Phase 6 + Data SNR Sweep)

1. Phase-4 `model_comparison_table.csv` exists as anchor.
2. Implement 2.1 first — unblocks Phase-5 checkbox in `docs/Tasks.md`.
3. Add one backbone at a time; each must run through existing NB05 scoring (no forked metric code).
