"""
Acoustic PDM — Evaluation, Anomaly Scoring & XAI Diagnostics Module
===================================================================
Scoring engines, threshold calibration (P95), SOTA metrics calculation
(ROC-AUC, pAUC @ 10% FPR, F1), and Explainable AI (XAI) difference heatmaps.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
from torch.utils.data import DataLoader

from .dataset import AcousticDataset


def compute_reconstruction_error(
    model: nn.Module,
    data: Union[np.ndarray, DataLoader],
    device: Optional[torch.device] = None,
    batch_size: int = 128
) -> np.ndarray:
    """
    Compute physical Mean Squared Error (MSE) reconstruction loss S_recon per block.
    
    MSE(X, X_hat) = (1 / 640) * sum_{f,t} (X_{f,t} - X_hat_{f,t})^2
    
    Args:
        model: Trained Autoencoder model.
        data: Array of shape (N, 1, 128, 5) or DataLoader.
        device: Torch device.
        batch_size: Batch size if converting array.
        
    Returns:
        1D numpy array of length N containing reconstruction errors.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    if isinstance(data, np.ndarray):
        dataset = AcousticDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        loader = data
        
    errors = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
            recon = model(x)
            # Per-block MSE across (1, 128, 5) -> (Batch,)
            diff = (x - recon) ** 2
            mse = diff.view(x.size(0), -1).mean(dim=1)
            errors.append(mse.cpu().numpy())
            
    return np.concatenate(errors, axis=0)


def compute_latent_score(
    model: nn.Module,
    if_model: object,
    data: Union[np.ndarray, DataLoader],
    device: Optional[torch.device] = None,
    batch_size: int = 128
) -> np.ndarray:
    """
    Compute Latent Outlier Score S_latent using Stage 2 Isolation Forest:
    S_latent = -score_samples(z)
    
    Args:
        model: Trained Conv2D-AE encoder.
        if_model: Fitted Latent Isolation Forest.
        data: Input blocks.
        
    Returns:
        1D numpy array of anomaly scores (higher = more anomalous).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    if isinstance(data, np.ndarray):
        dataset = AcousticDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        loader = data
        
    latents = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
            z = model.encode(x)
            latents.append(z.cpu().numpy())
            
    z_all = np.concatenate(latents, axis=0)
    # Higher score_samples means normal in sklearn, so invert for anomaly score
    latent_scores = -if_model.score_samples(z_all)
    return latent_scores


def compute_hybrid_score(
    s_recon: np.ndarray,
    s_latent: np.ndarray,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Linear fusion of Min-Max normalized physical and latent anomaly scores:
    S_final = alpha * S_recon_norm + (1 - alpha) * S_latent_norm
    
    Args:
        s_recon: Reconstruction MSE scores.
        s_latent: Latent outlier scores.
        alpha: Weight for reconstruction error (default 0.6).
        
    Returns:
        1D numpy array of fused anomaly scores in [0, 1].
    """
    def min_max(arr):
        denom = arr.max() - arr.min()
        return (arr - arr.min()) / (denom + 1e-10)
        
    norm_recon = min_max(s_recon)
    norm_latent = min_max(s_latent)
    
    return alpha * norm_recon + (1.0 - alpha) * norm_latent


def aggregate_blocks_to_clips(
    block_scores: np.ndarray,
    blocks_per_clip: int = 309,
    method: str = "mean"
) -> np.ndarray:
    """
    Aggregate per-block anomaly scores to per-clip (10s WAV) anomaly scores.
    Standard DCASE / MIMII SOTA evaluation protocol.
    
    Args:
        block_scores: 1D array of per-block anomaly scores.
        blocks_per_clip: Number of context blocks per 10s audio recording (~309).
        method: Aggregation method ('mean', 'top10', 'p90', 'max').
        
    Returns:
        1D numpy array of clip-level anomaly scores.
    """
    n_total = len(block_scores)
    n_clips = max(1, round(n_total / blocks_per_clip))
    clip_chunks = np.array_split(block_scores, n_clips)
    
    clip_scores = []
    for chunk in clip_chunks:
        if len(chunk) == 0:
            continue
        if method == "mean":
            clip_scores.append(float(np.mean(chunk)))
        elif method == "top10":
            k = max(1, int(np.ceil(len(chunk) * 0.10)))
            top_k = np.partition(chunk, -k)[-k:]
            clip_scores.append(float(np.mean(top_k)))
        elif method == "p90":
            clip_scores.append(float(np.percentile(chunk, 90)))
        elif method == "max":
            clip_scores.append(float(np.max(chunk)))
        else:
            clip_scores.append(float(np.mean(chunk)))
            
    return np.array(clip_scores, dtype=np.float32)


def calibrate_threshold(
    val_normal_scores: np.ndarray,
    percentile: float = 95.0
) -> float:
    """
    Calibrate diagnostic decision threshold theta on held-out normal validation data:
    theta = P_95 (guarantees <= 5% false positive rate on normal operational sound).
    
    Args:
        val_normal_scores: Anomaly scores on normal validation clips.
        percentile: Percentile threshold (default 95.0).
        
    Returns:
        Float threshold value theta.
    """
    return float(np.percentile(val_normal_scores, percentile))


def compute_sota_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate SOTA acoustic anomaly detection benchmark metrics:
    - ROC-AUC: Area under the Receiver Operating Characteristic curve.
    - pAUC: Partial AUC with max false-alarm rate FPR <= 10% (crucial for industrial PdM).
    - Precision, Recall, F1 Score at calibrated threshold.
    """
    auc = roc_auc_score(y_true, y_scores)
    
    # Partial AUC at max_fpr = 0.1
    pauc = roc_auc_score(y_true, y_scores, max_fpr=0.1)
    
    metrics = {
        "roc_auc": float(auc),
        "pauc_10": float(pauc),
    }
    
    if threshold is not None:
        y_pred = (y_scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        metrics.update({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        })
        
    return metrics


def compute_xai_difference_heatmap(
    original_block: np.ndarray,
    reconstructed_block: np.ndarray
) -> np.ndarray:
    """
    Generate Explainable AI (XAI) difference spectrogram heatmap:
    D = |X - X_hat|
    
    Pinpoints the exact frequency bands and temporal instants responsible for the anomaly.
    
    Args:
        original_block: Input 2D patch of shape (128, 5) or (1, 128, 5).
        reconstructed_block: Reconstructed 2D patch of same shape.
        
    Returns:
        2D numpy array of shape (128, 5) with absolute difference values.
    """
    orig = original_block.squeeze()
    recon = reconstructed_block.squeeze()
    return np.abs(orig - recon)
