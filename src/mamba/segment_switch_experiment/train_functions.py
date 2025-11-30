"""
Training functions for segment switch experiment.

Optimized for pre-computed tensor data with fixed sequence lengths.
"""

import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Any, Optional


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    verbose: bool = False
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        verbose: Whether to print batch progress
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    total_batches = len(dataloader)
    epoch_start = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()
        
        # Move data to device
        X = batch['X'].to(device)  # (B, T, 2)
        dt = batch['dt'].to(device)  # (B, T)
        
        seq_len = X.shape[1]
        if seq_len < 2:
            continue
        
        # Input: all but last timestep
        X_input = X[:, :-1, :]  # (B, T-1, 2)
        # Target: all but first timestep (next positions)
        Y_target = X[:, 1:, :]  # (B, T-1, 2)
        dt_input = dt[:, :-1].unsqueeze(-1)  # (B, T-1, 1)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_input, dt_input)  # (B, T-1, 2)
        
        # Compute loss (L1)
        loss = criterion(predictions, Y_target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        batch_time = time.time() - batch_start
        if verbose and (batch_idx % 50 == 0 or batch_idx == total_batches - 1):
            elapsed = time.time() - epoch_start
            eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1)
            print(f"  Batch {batch_idx + 1}/{total_batches} | Loss: {loss.item():.6f} | "
                  f"Time: {batch_time:.2f}s | ETA: {eta/60:.1f}min")

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    verbose: bool = False
) -> float:
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to use
        verbose: Whether to print batch progress
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if verbose and batch_idx % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{total_batches}")
            
            X = batch['X'].to(device)
            dt = batch['dt'].to(device)
            
            seq_len = X.shape[1]
            if seq_len < 2:
                continue
            
            X_input = X[:, :-1, :]
            Y_target = X[:, 1:, :]
            dt_input = dt[:, :-1].unsqueeze(-1)
            
            predictions = model(X_input, dt_input)
            loss = criterion(predictions, Y_target)
            
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def train_epoch_fp16(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_accum_steps: int = 1,
    verbose: bool = False
) -> float:
    """
    Train for one epoch with optional mixed precision (FP16) and gradient accumulation.
    
    Args:
        model: The model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        scaler: GradScaler for FP16 training (None for FP32)
        grad_accum_steps: Number of gradient accumulation steps
        verbose: Whether to print batch progress
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    total_batches = len(dataloader)
    epoch_start = time.time()
    
    use_fp16 = scaler is not None

    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()
        
        # Move data to device
        X = batch['X'].to(device)  # (B, T, 2)
        dt = batch['dt'].to(device)  # (B, T)
        
        seq_len = X.shape[1]
        if seq_len < 2:
            continue
        
        # Input: all but last timestep
        X_input = X[:, :-1, :]  # (B, T-1, 2)
        # Target: all but first timestep (next positions)
        Y_target = X[:, 1:, :]  # (B, T-1, 2)
        dt_input = dt[:, :-1].unsqueeze(-1)  # (B, T-1, 1)
        
        # Forward pass with optional autocast
        if use_fp16:
            with torch.cuda.amp.autocast():
                predictions = model(X_input, dt_input)
                loss = criterion(predictions, Y_target)
                loss = loss / grad_accum_steps  # Scale for accumulation
        else:
            predictions = model(X_input, dt_input)
            loss = criterion(predictions, Y_target)
            loss = loss / grad_accum_steps
        
        # Backward pass
        if use_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps  # Unscale for logging
        num_batches += 1
        
        batch_time = time.time() - batch_start
        elapsed = time.time() - epoch_start
        eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1)
        if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
            print(f"  Batch {batch_idx + 1}/{total_batches} | Loss: {loss.item()*grad_accum_steps:.6f} | "
                  f"Time: {batch_time:.2f}s | ETA: {eta/60:.1f}min")
    
    # Handle remaining gradients
    if num_batches % grad_accum_steps != 0:
        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_fp16(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_fp16: bool = False,
    verbose: bool = False
) -> float:
    """
    Validate the model with optional mixed precision.
    
    Args:
        model: The model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to use
        use_fp16: Whether to use FP16 inference
        verbose: Whether to print batch progress
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if verbose and batch_idx % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{total_batches}")
            
            X = batch['X'].to(device)
            dt = batch['dt'].to(device)
            
            seq_len = X.shape[1]
            if seq_len < 2:
                continue
            
            X_input = X[:, :-1, :]
            Y_target = X[:, 1:, :]
            dt_input = dt[:, :-1].unsqueeze(-1)
            
            if use_fp16:
                with torch.cuda.amp.autocast():
                    predictions = model(X_input, dt_input)
                    loss = criterion(predictions, Y_target)
            else:
                predictions = model(X_input, dt_input)
                loss = criterion(predictions, Y_target)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, List]:
    """
    Generate predictions for all trajectories in the dataloader.
    
    Args:
        model: The trained model
        dataloader: DataLoader to get predictions for
        device: Device to use
    
    Returns:
        Dictionary with:
        - preds: List of predicted trajectories (un-normalized)
        - targets: List of target trajectories (un-normalized)
        - real_positions: List of clean trajectories
        - measurement_noise_std: List of noise std values
    """
    model.eval()
    
    results = {
        "preds": [],
        "targets": [],
        "real_positions": [],
        "measurement_noise_std": []
    }
    
    total_batches = len(dataloader)
    print("Generating predictions...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{total_batches}")
            
            X = batch['X'].to(device)
            Y = batch['Y'].to(device)
            dt = batch['dt'].to(device)
            X_mean = batch['X_mean'].to(device)  # (B, 1, 2)
            X_std = batch['X_std'].to(device)    # (B, 1, 2)
            meas_noise = batch['measurement_noise_std']  # (B, 2)
            
            # Run model
            X_input = X[:, :-1, :]
            dt_input = dt[:, :-1].unsqueeze(-1)
            preds_norm = model(X_input, dt_input)
            
            # Un-normalize predictions and targets
            preds_real = (preds_norm * X_std) + X_mean
            targets_real = (X[:, 1:, :] * X_std) + X_mean
            
            # Convert to numpy
            preds_np = preds_real.cpu().numpy()
            targets_np = targets_real.cpu().numpy()
            Y_np = Y[:, 1:, :].cpu().numpy()  # Align with predictions
            meas_np = meas_noise.cpu().numpy()
            
            # Store results
            batch_size = preds_np.shape[0]
            for i in range(batch_size):
                results["preds"].append(preds_np[i])
                results["targets"].append(targets_np[i])
                results["real_positions"].append(Y_np[i])
                results["measurement_noise_std"].append(meas_np[i])
    
    print(f"Generated {len(results['preds'])} predictions.")
    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_path: str,
    config: Dict[str, Any]
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device
) -> tuple:
    """
    Load model checkpoint.
    
    Returns:
        (start_epoch, train_losses, val_losses)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    train_loss = checkpoint.get('train_loss', None)
    val_loss = checkpoint.get('val_loss', None)
    
    # Handle both single value and list formats
    train_losses = [train_loss] if train_loss is not None and not isinstance(train_loss, list) else (train_loss or [])
    val_losses = [val_loss] if val_loss is not None and not isinstance(val_loss, list) else (val_loss or [])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  Resuming from epoch {start_epoch}")
    
    return start_epoch, train_losses, val_losses
