"""
Acoustic PDM — Preprocessing Module
====================================
Digital Signal Processing (DSP) functions for transforming 1D raw audio waveforms
into normalized 2D log Mel-spectrogram context blocks for neural network consumption.

Dataset Scope: Hitachi MIMII (Fan, Pump, Slider, Valve @ id_00, 6 dB SNR).
Zero Data Leakage: Normalization statistics (mu, sigma) are computed strictly on
training normal data.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import librosa
import soundfile as sf
import yaml


def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> dict:
    """Load centralized pipeline configuration from YAML file."""
    p = Path(config_path)
    if not p.exists():
        # Search upward for configs/config.yaml
        for parent in [Path("."), Path(".."), Path("../..")]:
            candidate = parent / "configs" / "config.yaml"
            if candidate.exists():
                p = candidate
                break
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def load_audio(
    file_path: Union[str, Path],
    target_sr: int = 16000,
    mono: bool = True
) -> np.ndarray:
    """
    Load a WAV audio file, enforce target sample rate (16 kHz) and mono channel.
    
    Args:
        file_path: Path to .wav file.
        target_sr: Target sample rate in Hz (default 16000).
        mono: Whether to convert multi-channel audio to mono (channel 0).
        
    Returns:
        1D float32 numpy array of audio samples.
    """
    path_str = str(file_path)
    try:
        audio, sr = sf.read(path_str, dtype="float32", always_2d=True)
        # Select first microphone channel (channel 0 of 8-mic array)
        if mono and audio.shape[1] > 1:
            audio = audio[:, 0]
        else:
            audio = audio.squeeze()
            
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
        return audio
    except Exception:
        # Fallback to librosa.load
        audio, _ = librosa.load(path_str, sr=target_sr, mono=mono)
        return audio.astype(np.float32)


def wav_to_log_mel(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power: float = 2.0,
    top_db: float = 80.0,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Convert 1D audio waveform to 2D log Mel-spectrogram (dB scale).
    
    Mathematical Pipeline:
        1. STFT: Linear spectrogram (513 bins x T frames)
        2. Mel-Filterbank: 128 non-linear frequency bins (0 to sr/2)
        3. Power to dB: S_dB = 10 * log10(Power + eps)
    
    Args:
        audio: 1D audio array.
        sr: Sample rate (16000 Hz).
        n_fft: FFT window length (1024 samples = 64 ms).
        hop_length: Hop length (512 samples = 32 ms, 50% overlap).
        n_mels: Number of Mel frequency bins (128).
        fmin: Minimum frequency (0 Hz).
        fmax: Maximum frequency (None = sr/2 = 8000 Hz).
        power: Exponent for magnitude spectrogram (2.0 = power).
        top_db: Dynamic range threshold in dB (80.0).
        epsilon: Numerical guard to prevent log(0).
        
    Returns:
        2D float32 numpy array of shape (n_mels, time_frames).
    """
    # Compute Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power
    )
    
    # Numerical guard & power-to-dB conversion
    mel_spec = np.maximum(mel_spec, epsilon)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=top_db)
    
    return log_mel.astype(np.float32)


def extract_context_blocks(
    log_mel: np.ndarray,
    context_frames: int = 5,
    channel_dim: bool = True
) -> np.ndarray:
    """
    Slice a 2D log Mel-spectrogram into consecutive overlapping context window blocks.
    
    Each 10s recording (313 frames) produces N = 313 - 5 + 1 = 309 blocks (or 305-309).
    Block shape: (n_mels, context_frames) = (128, 5) spanning 160 ms temporal context.
    
    Args:
        log_mel: 2D array of shape (n_mels, time_frames).
        context_frames: Number of consecutive time frames (default 5).
        channel_dim: If True, adds channel dimension -> (N, 1, 128, 5).
        
    Returns:
        3D or 4D numpy array of blocks: shape (N, 1, 128, 5) or (N, 128, 5).
    """
    n_mels, total_frames = log_mel.shape
    if total_frames < context_frames:
        raise ValueError(f"Spectrogram frames ({total_frames}) < context_frames ({context_frames})")
        
    num_blocks = total_frames - context_frames + 1
    blocks = np.empty((num_blocks, n_mels, context_frames), dtype=np.float32)
    
    for i in range(num_blocks):
        blocks[i] = log_mel[:, i : i + context_frames]
        
    if channel_dim:
        blocks = np.expand_dims(blocks, axis=1)  # (N, 1, 128, 5)
        
    return blocks


def compute_norm_stats(
    train_blocks: np.ndarray,
    epsilon: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-Mel-frequency-bin mean (mu) and standard deviation (sigma)
    EXCLUSIVELY on normal training blocks (Zero Data Leakage guarantee).
    
    Args:
        train_blocks: Array of shape (N, 1, 128, 5) or (N, 128, 5).
        epsilon: Numerical guard added to std to prevent division by zero.
        
    Returns:
        (mean, std) tuple: each of shape (128,).
    """
    # If 4D (N, 1, 128, 5), squeeze channel -> (N, 128, 5)
    data = train_blocks.squeeze(1) if train_blocks.ndim == 4 else train_blocks
    
    # Compute stats across all blocks (axis 0) and context frames (axis 2) -> per-bin (128,)
    mean = np.mean(data, axis=(0, 2)).astype(np.float32)
    std = np.std(data, axis=(0, 2)).astype(np.float32)
    
    # Ensure std is strictly positive
    std = np.maximum(std, epsilon)
    
    return mean, std


def apply_zscore_norm(
    blocks: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Apply frozen Z-score normalization: X_norm = (X - mu) / (sigma + eps).
    
    Args:
        blocks: Array of shape (N, 1, 128, 5) or (N, 128, 5).
        mean: Per-frequency-bin mean vector of shape (128,).
        std: Per-frequency-bin std vector of shape (128,).
        epsilon: Numerical guard.
        
    Returns:
        Normalized numpy array of the same shape as input.
    """
    has_channel = (blocks.ndim == 4)
    if has_channel:
        # Reshape mean and std to broadcast over (N, 1, 128, 5) -> (1, 1, 128, 1)
        mu = mean.reshape(1, 1, -1, 1)
        sigma = std.reshape(1, 1, -1, 1)
    else:
        # Reshape to (1, 128, 1)
        mu = mean.reshape(1, -1, 1)
        sigma = std.reshape(1, -1, 1)
        
    normalized = (blocks - mu) / (sigma + epsilon)
    return normalized.astype(np.float32)


def process_single_audio_file(
    file_path: Union[str, Path],
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    config: Optional[dict] = None
) -> np.ndarray:
    """
    End-to-end DSP processing for a single audio file:
    WAV -> Resample (16kHz) -> Log-Mel (128 bins) -> Framing (128, 5) -> Z-Score.
    
    Used directly during inference and live web app diagnostics.
    
    Args:
        file_path: Path to input .wav file.
        mean: Optional precomputed mean for normalization.
        std: Optional precomputed std for normalization.
        config: Configuration dictionary (uses defaults if None).
        
    Returns:
        Normalized tensor blocks of shape (N, 1, 128, 5).
    """
    cfg = config or load_config()
    audio_cfg = cfg.get("audio", {})
    feat_cfg = cfg.get("features", {})
    norm_cfg = cfg.get("normalization", {})
    
    sr = audio_cfg.get("sample_rate", 16000)
    n_fft = feat_cfg.get("n_fft", 1024)
    hop_length = feat_cfg.get("hop_length", 512)
    n_mels = feat_cfg.get("n_mels", 128)
    context_frames = feat_cfg.get("context_frames", 5)
    eps = norm_cfg.get("epsilon", 1e-8)
    
    # 1. Load audio
    audio = load_audio(file_path, target_sr=sr, mono=True)
    
    # 2. Extract log Mel-spectrogram
    log_mel = wav_to_log_mel(
        audio=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # 3. Context framing
    blocks = extract_context_blocks(log_mel, context_frames=context_frames, channel_dim=True)
    
    # 4. Z-Score normalization if stats provided
    if mean is not None and std is not None:
        blocks = apply_zscore_norm(blocks, mean=mean, std=std, epsilon=eps)
        
    return blocks
