"""
Acoustic PDM — Model Training & Dual-Stage Hybrid Engine
========================================================
Training routines for PyTorch Autoencoders, Shallow Baselines,
and Dual-Stage Deep Hybrid (Conv2D-AE + Latent Isolation Forest).
"""

import copy
import os
from pathlib import Path
import time
from typing import Dict, Optional, Tuple, Union
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import xgboost as xgb

from .dataset import AcousticDataset, get_dataloaders
from .model_ae import Conv2DAutoencoder, FCAutoencoder, LSTMAutoencoder


def get_device() -> torch.device:
    """Auto-detect available accelerator (CUDA GPU, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 10,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train an Autoencoder model using MSE loss and Adam optimizer with early stopping.
    
    Args:
        model: PyTorch Autoencoder instance.
        train_loader: DataLoader for normal training data.
        val_loader: Optional DataLoader for held-out normal validation data.
        epochs: Maximum training epochs (default 50).
        lr: Initial learning rate (default 0.001).
        weight_decay: L2 regularization weight decay (default 1e-5).
        patience: Early stopping patience epochs (default 10).
        device: Torch device (auto-detected if None).
        checkpoint_path: Optional path to save the best model weights (.pth).
        verbose: Whether to print progress per epoch.
        
    Returns:
        (best_model, history_dict)
    """
    device = device or get_device()
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, verbose=verbose
    )
    
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # ── Training ──
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        
        for batch in train_loader:
            if isinstance(batch, (tuple, list)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
                
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * x.size(0)
            train_batches += x.size(0)
            
        train_loss = train_loss_sum / max(1, train_batches)
        history["train_loss"].append(train_loss)
        
        # ── Validation ──
        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (tuple, list)):
                        x = batch[0].to(device)
                    else:
                        x = batch.to(device)
                    recon = model(x)
                    loss = criterion(recon, x)
                    val_loss_sum += loss.item() * x.size(0)
                    val_batches += x.size(0)
            val_loss = val_loss_sum / max(1, val_batches)
            scheduler.step(val_loss)
            
        history["val_loss"].append(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)
        
        if verbose:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] — Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
            
        # ── Early Stopping & Checkpoint ──
        monitor_loss = val_loss if val_loader is not None else train_loss
        if monitor_loss < best_val_loss:
            best_val_loss = monitor_loss
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
            if checkpoint_path:
                os.makedirs(os.path.dirname(str(checkpoint_path)), exist_ok=True)
                torch.save(best_weights, str(checkpoint_path))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"⏹ Early stopping triggered after {epoch} epochs (Best Loss: {best_val_loss:.6f})")
                break
                
    elapsed = time.time() - start_time
    if verbose:
        print(f"✓ Training finished in {elapsed:.1f}s. Best Loss: {best_val_loss:.6f}")
        
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model, history


def extract_latent_vectors(
    model: nn.Module,
    data_or_loader: Union[np.ndarray, DataLoader],
    batch_size: int = 128,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Extract 32-dimensional bottleneck latent vectors z from a trained Autoencoder.
    
    Args:
        model: Trained Autoencoder model.
        data_or_loader: NumPy array of blocks or PyTorch DataLoader.
        batch_size: Batch size if converting numpy array.
        device: Torch device.
        
    Returns:
        2D numpy array of shape (N, 32).
    """
    device = device or get_device()
    model = model.to(device)
    model.eval()
    
    if isinstance(data_or_loader, np.ndarray):
        dataset = AcousticDataset(data_or_loader)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        loader = data_or_loader
        
    latents = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
            z = model.encode(x)
            latents.append(z.cpu().numpy())
            
    return np.concatenate(latents, axis=0)


def train_latent_isolation_forest(
    latent_train: np.ndarray,
    n_estimators: int = 100,
    contamination: float = 0.05,
    random_state: int = 42,
    save_path: Optional[Union[str, Path]] = None
) -> IsolationForest:
    """
    Fit an Isolation Forest on the 32-dimensional normal latent manifold (Stage 2 of Hybrid).
    
    Args:
        latent_train: Latent vectors of shape (N, 32) from training normal data.
        n_estimators: Number of isolation trees (default 100).
        contamination: Expected anomaly fraction (default 0.05).
        random_state: Random seed (42).
        save_path: Path to save .joblib model.
        
    Returns:
        Fitted IsolationForest model.
    """
    # Subsample if large to fit cleanly within memory
    max_samples = min(len(latent_train), 50000)
    if len(latent_train) > max_samples:
        idx = np.random.RandomState(random_state).choice(len(latent_train), max_samples, replace=False)
        fit_data = latent_train[idx]
    else:
        fit_data = latent_train
        
    if_model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    if_model.fit(fit_data)
    
    if save_path:
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        joblib.dump(if_model, str(save_path))
        
    return if_model


def train_shallow_baselines(
    train_normal: np.ndarray,
    val_anomaly: Optional[np.ndarray] = None,
    random_state: int = 42,
    save_dir: Optional[Union[str, Path]] = None
) -> Dict[str, object]:
    """
    Train benchmark baseline models on flattened 640-dim DSP features:
    1. Isolation Forest (Shallow)
    2. One-Class SVM (RBF kernel)
    3. XGBoost Classifier (Supervised trap baseline)
    """
    # Flatten (N, 1, 128, 5) -> (N, 640)
    flat_train = train_normal.reshape(len(train_normal), -1)
    
    # Subsample for fast fitting of OCSVM/IF
    max_n = min(len(flat_train), 25000)
    idx = np.random.RandomState(random_state).choice(len(flat_train), max_n, replace=False)
    sub_train = flat_train[idx]
    
    baselines = {}
    
    # 1. Isolation Forest (Shallow)
    if_shallow = IsolationForest(n_estimators=100, contamination=0.05, random_state=random_state, n_jobs=-1)
    if_shallow.fit(sub_train)
    baselines["isolation_forest_shallow"] = if_shallow
    
    # 2. One-Class SVM (RBF)
    ocsvm_sub = sub_train[:min(5000, len(sub_train))]
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(ocsvm_sub)
    baselines["one_class_svm_shallow"] = ocsvm
    
    # 3. XGBoost Supervised (Train on normal + synthetic/known anomaly to demonstrate failure trap)
    if val_anomaly is not None:
        flat_anom = val_anomaly.reshape(len(val_anomaly), -1)[:min(len(sub_train), 5000)]
        X_sup = np.vstack([sub_train[:len(flat_anom)], flat_anom])
        y_sup = np.array([0] * len(flat_anom) + [1] * len(flat_anom))
        
        xgb_clf = xgb.XGBClassifier(
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss"
        )
        xgb_clf.fit(X_sup, y_sup)
        baselines["xgboost_supervised"] = xgb_clf
        
    if save_dir:
        os.makedirs(str(save_dir), exist_ok=True)
        for name, model in baselines.items():
            joblib.dump(model, os.path.join(str(save_dir), f"{name}.joblib"))
            
    return baselines
