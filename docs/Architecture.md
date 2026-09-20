# System Architecture & Technical Specifications
> **End-to-end DSP, deep learning pipeline, inference engines, and modular notebook dependency DAG.**

---

## 1. End-to-End DSP Preprocessing Pipeline

The mathematical transformation from 1D sound pressure waves to normalized 2D neural network input tensors:

```mermaid
flowchart TD
    WAV["Raw Audio File (.wav)\n10s @ 16 kHz mono (160,000 samples)"] --> STFT["1. STFT\nN_fft = 1024 (64 ms), Hop = 512 (32 ms, 50% overlap)\nLinear Spectrogram (513 bins x 309 frames)"]
    
    STFT --> MEL["2. Mel-Filterbank (128 Bins)\nNon-linear perceptual frequency scale (0 - 8000 Hz)\nPower Spectrogram (128 bins x 309 frames)"]
    
    MEL --> DB["3. Log-Decibel Scaling\nS_dB = 10 * log10(Power + 1e-10)\nCompresses 96 dB dynamic range"]
    
    DB --> FRAME["4. Context Framing (P = 5 frames)\n305 consecutive blocks of shape (128, 5)\nCaptures temporal dynamics (640 dims per block)"]
    
    FRAME --> SPLIT["5. Clip-Level Train/Validation Split (90/10)\nSplit whole recordings before framing (prevents leakage)"]
    
    SPLIT --> ZNORM["6. Leak-Free Z-Score Normalization\nX_norm = (X - mu_train) / (sigma_train + 1e-8)\nComputed EXCLUSIVELY on training normal data"]
    
    ZNORM --> OUT["Saved Tensors (.npy) & norm_stats.json\nReady for model training and inference"]
```

---

## 2. Model Architectures

### 2.1 Convolutional 2D Autoencoder (Conv2D-AE)
- **Input:** 2D tensor of shape `(Batch, 1, 128, 5)`.
- **Encoder:** 3 Conv2D layers with LeakyReLU activations and stride 2 downsampling (channels: 1 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 128), flattened to a 32-dimensional bottleneck $z \in \mathbb{R}^{32}$.
- **Decoder:** Symmetrical ConvTranspose2D layers reconstructing original shape `(Batch, 1, 128, 5)`.
- **Loss:** Mean Squared Error (MSE) over normal training blocks:
  $$\mathcal{L}_{\text{MSE}}(X, \hat{X}) = \frac{1}{B \cdot 128 \cdot 5} \sum_{b,f,t} (X_{b,f,t} - \hat{X}_{b,f,t})^2$$

### 2.2 Dual-Stage Deep Hybrid Model (Conv2D-AE + Latent Isolation Forest)

```mermaid
flowchart TD
    IN["Input Block (1, 128, 5)"] --> ENC["Stage 1: Conv2D-AE Encoder"]
    ENC --> Z["Latent Vector (z in R^32)"]
    
    Z --> DEC["Conv2D-AE Decoder\n(Reconstruction)"]
    Z --> IF["Stage 2: Latent Isolation Forest\n(Density Estimation)"]
    
    DEC --> S_REC["Physical Reconstruction Loss S_recon\nMSE(X, X̂)"]
    IF --> S_LAT["Latent Outlier Score S_latent\n-score_samples(z)"]
    
    S_REC --> NORM["Min-Max Score Normalization"]
    S_LAT --> NORM
    
    NORM --> FUSION["Linear Fusion: S_final = 0.6 · S̃_recon + 0.4 · S̃_latent"]
    FUSION --> DECISION["Diagnostic Decision: Healthy vs. Faulty\nPeak SOTA Performance (~97% ROC-AUC)"]
```

---

## 3. Real-Time Inference & Decision Pipeline

During inference, an unknown 10-second audio clip undergoes framing into $N = 305$ blocks:

```mermaid
flowchart TD
    WAV["Incoming Machine Audio (.wav)"] --> DSP["DSP Extractor (STFT ──► Mel ──► dB ──► Z-Score)"]
    DSP --> SLICE["Framing into N = 305 Blocks (128 x 5)"]
    SLICE --> AE["Conv2D-AE + Latent IF Model"]
    AE --> FUSION["Block Hybrid Anomaly Scoring"]
    FUSION --> AGG["Clip-Level Mean Aggregation: A(y) = (1/N) sum(s_n)"]
    AGG --> COMP{"A(y) > theta ?"}
    
    COMP -- "No (A(y) <= theta)" --> HEALTHY["Status: HEALTHY (NORMAL) 🟢\nAsset operating within normal bounds"]
    COMP -- "Yes (A(y) > theta)" --> FAULT["Status: FAULT DETECTED 🔴\nFlagged for maintenance inspection"]
    
    FAULT --> XAI["Difference Spectrogram Heatmap: D = |X - X̂|\nPinpoints exact defect frequency band"]
```

### Threshold Selection ($\theta$):
- **Percentile Calibration (Primary):** $\theta = P_{95}$ on held-out normal validation clips (guarantees $\le 5\%$ false alarm rate).
- **Gaussian Parametric:** $\theta = \mu_{\text{val}} + 3 \cdot \sigma_{\text{val}}$ (covers 99.73% of normal variation).

---

## 4. Chained Kaggle Notebooks (Dependency DAG)

```mermaid
flowchart TD
    DSET["Kaggle Dataset: daisukelab/dc2020task2"] --> NB01["NB01: Data Acquisition & Inspection\nOutput: reports/indexed_dataset.csv"]
    
    NB01 --> NB02["NB02: Audio Exploration & Visuals\n(Standalone EDA: Waveforms & Heatmaps)"]
    NB01 --> NB03["NB03: Preprocessing Engine\nOutput: data/processed/*.npy\nreports/norm_stats.json"]
    
    NB03 --> NB04["NB04: Model Training & Baselines\nOutput: models/best_ae_*.pth\nmodels/hybrid_iforest_*.joblib"]
    
    NB03 --> NB05["NB05: Evaluation & Metrics\nOutput: reports/model_comparison_table.csv\nreports/multi_model_roc_curves.png"]
    NB04 --> NB05
    
    NB04 --> NB06["NB06: Interactive Diagnostic App\n(Gradio / Streamlit Cloudflare Public URL)"]
```

### Kaggle Input Attachment Rules:
- **NB01:** Attach `daisukelab/dc2020task2`.
- **NB02:** Attach `daisukelab/dc2020task2` + NB01 output (exploratory visualization only).
- **NB03:** Attach `daisukelab/dc2020task2` + NB01 output.
- **NB04:** Attach NB03 output (`data/processed/`).
- **NB05:** Attach NB03 output + NB04 output.
- **NB06:** Attach NB04 output (`models/`).

---

## 5. Directory Layout & Module Structure

```
Acoustic_pdm/
├── Agent.md                                      # AI agent operating rules
├── docs/                                         # Unified project documentation
│   ├── Project.md                                # Project summary, value proposition & scope
│   ├── Architecture.md                           # DSP pipeline, ML architectures, inference & DAG
│   ├── Memory.md                                 # Technical decisions, blockers & lessons learned
│   └── Tasks.md                                  # Phase-by-phase task breakdown & progress tracker
├── configs/
│   └── config.yaml                               # Centralized hyperparameters for entire pipeline
├── data/
│   ├── raw/                                      # Local sample .wav files (gitignored)
│   └── processed/                                # Preprocessed .npy tensors (gitignored)
├── models/                                       # Saved model checkpoints (gitignored)
├── notebooks/
│   ├── 01-data-acquisition-and-inspection.ipynb  # Phase 1: Dataset catalog & audit
│   ├── 02-audio-exploration-and-visualization.ipynb # Phase 1: EDA waveforms & spectrograms
│   ├── 03-preprocessing-engine.ipynb            # Phase 2: Spectrogram extraction & framing
│   ├── 04-model-training-and-baselines.ipynb     # Phase 3: Standalone baselines & hybrid training
│   ├── 05-evaluation-and-metrics.ipynb           # Phase 4: Comparative evaluation & SOTA benchmarking
│   └── 06-interactive-diagnostic-app.ipynb       # Phase 6: Cloud-hosted live web demo
├── reports/                                      # Exported metrics, CSV tables, and ROC plots
├── src/
│   ├── __init__.py
│   ├── preprocess.py                             # Audio loading, STFT, Mel-filterbank, framing
│   ├── dataset.py                                # PyTorch Dataset & DataLoader
│   ├── model_ae.py                               # Conv2D-AE, FC-AE, LSTM-AE architectures
│   ├── model_vae.py                              # Variational Autoencoder (Stage B)
│   ├── train.py                                  # Training loop with early stopping & hybrid fitting
│   └── evaluate.py                               # Scoring, thresholding, and metric computation
├── app.py                                        # Local offline Streamlit web demo
├── requirements.txt                              # Pinned Python dependencies
└── .gitignore                                    # Clean version control exclusions
```
