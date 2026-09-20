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

5. **Dual-Platform Strategy & Cloud Demo:**
   - 100% of compute-heavy tasks execute on Kaggle Cloud (P100/T4 GPUs).
   - Local persistence holds `.npy` arrays, model checkpoints (`models/`), and reports (`reports/`) for offline execution.
   - Notebook 06 launches Gradio with `share=True` or Streamlit with Cloudflare Tunnel, generating a public URL for viva presentation on any mobile/laptop browser.
