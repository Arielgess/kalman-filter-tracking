"""
Training script for Mamba model on segment switch experiment.

Uses pre-computed tensor dataset for fast training.

Usage:
    python train_mamba.py [--num_epochs N] [--batch_size B] [--from_scratch]
    
Arguments:
    --num_epochs: Number of epochs to train (default: 50)
    --batch_size: Batch size (default: 64)
    --from_scratch: Train from scratch instead of loading checkpoint
    --checkpoint_path: Path to checkpoint to resume from
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Add workspace root to path
script_dir = Path(__file__).parent
workspace_root = script_dir.parent.parent.parent
sys.path.insert(0, str(workspace_root))

from src.mamba.mamba_model import TrajectoryMambaModel, ModelArgs
from src.mamba.segment_switch_experiment.tensor_dataloader import (
    TensorTrajectoryDataset, tensor_collate_fn
)
from src.mamba.segment_switch_experiment.train_functions import (
    train_epoch, validate, save_checkpoint, load_checkpoint
)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Mamba model on segment switch data')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default 16 for 6GB GPU)')
    parser.add_argument('--from_scratch', action='store_true', help='Train from scratch')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Configuration
    config = {
        # Data
        'tensor_path': str(script_dir / 'data' / 'tensor_dataset.pt'),
        'train_split': 0.9,
        'val_split': 0.1,
        
        # Data Dimensions
        'input_dim': 2,
        
        # Model Architecture (reduced for 6GB GPU)
        'd_model': 64,       # Reduced for memory
        'n_layer': 4,         # Reduced for memory
        'd_state': 16,        # Reduced for memory
        'expand': 2,
        
        # Training
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'weight_decay': 0.01,
        'early_stopping_patience': 15,
        
        # Checkpointing
        'checkpoint_dir': str(script_dir / 'checkpoints'),
        
        # Other
        'num_workers': 0,  # Windows compatibility
        'seed': 42
    }
    
    # Memory optimization tips
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem < 8:
            print(f"\n⚠️  GPU has only {gpu_mem:.1f}GB - using memory-optimized settings")
            print(f"   Batch size: {config['batch_size']}, Model: d={config['d_model']}, layers={config['n_layer']}")
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    print("\n" + "=" * 60)
    print("SEGMENT SWITCH EXPERIMENT - MAMBA TRAINING")
    print("=" * 60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Check tensor file exists
    tensor_path = Path(config['tensor_path'])
    if not tensor_path.exists():
        print(f"\n❌ Tensor dataset not found: {tensor_path}")
        print("   Run 'python convert_to_tensor.py' first to create it.")
        sys.exit(1)
    
    # =========================================================================
    # RIGOROUS TRAINING-SET NORMALIZATION (Prevents Data Leakage)
    # =========================================================================
    print("\n" + "-" * 60)
    print("Loading data with TRAINING-SET normalization (no data leakage)")
    print("-" * 60)
    
    # Step 1: Load raw data first (before creating dataset)
    print("\nStep 1: Loading raw tensor data...")
    raw_data = torch.load(tensor_path)
    raw_X = raw_data['X']  # (N, T, 2)
    n_trajectories = raw_X.shape[0]
    dim = raw_X.shape[-1]
    print(f"  Total trajectories: {n_trajectories}")
    print(f"  Sequence length: {raw_X.shape[1]}")
    print(f"  Dimensions: {dim}")
    
    # Step 2: Generate split indices with shuffling
    print("\nStep 2: Generating train/val split indices...")
    train_size = int(config['train_split'] * n_trajectories)
    val_size = n_trajectories - train_size
    
    # Create shuffled indices using seeded RNG
    rng = torch.Generator().manual_seed(config['seed'])
    shuffled_indices = torch.randperm(n_trajectories, generator=rng).tolist()
    
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]
    
    print(f"  Train indices: {train_size} trajectories")
    print(f"  Val indices: {val_size} trajectories")
    
    # Step 3: Compute statistics on TRAINING DATA ONLY
    print("\nStep 3: Computing normalization stats from TRAINING data only...")
    train_X = raw_X[train_indices]  # (train_size, T, 2)
    
    # Global mean/std across batch (0) and time (1) dimensions, keeping feature dim (2)
    train_mean = train_X.mean(dim=(0, 1))  # (2,)
    train_std = train_X.std(dim=(0, 1))    # (2,)
    
    # Add epsilon to prevent division by zero
    eps = 1e-8
    train_std = torch.where(train_std < eps, torch.ones_like(train_std), train_std)
    
    print(f"  Training mean: {train_mean.tolist()}")
    print(f"  Training std: {train_std.tolist()}")
    
    # Create stats dictionary for dataset
    training_stats = {
        'mean': train_mean,
        'std': train_std
    }
    
    # Clean up raw data to free memory
    del raw_data, raw_X, train_X
    
    # Step 4: Initialize dataset with TRAINING stats
    print("\nStep 4: Initializing dataset with training stats...")
    dataset = TensorTrajectoryDataset(
        str(tensor_path), 
        normalize=True, 
        stats=training_stats  # Force use of training-only stats
    )
    
    # Step 5: Create Subsets using pre-computed indices
    print("\nStep 5: Creating train/val subsets...")
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"  Train subset: {len(train_dataset)} trajectories")
    print(f"  Val subset: {len(val_dataset)} trajectories")
    
    # Step 6: Create DataLoaders
    print("\nStep 6: Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=tensor_collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=tensor_collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    # Save training stats to config for checkpoint (enables inference later)
    config['global_stats'] = {
        'mean': training_stats['mean'].tolist(),
        'std': training_stats['std'].tolist()
    }
    config['train_indices'] = train_indices
    config['val_indices'] = val_indices
    
    print("\n✓ Data loading complete (no data leakage)")
    print(f"  Stats computed from: {train_size} training trajectories only")
    print(f"  Stats applied to: {n_trajectories} total trajectories")
    
    # Create model
    print("\n" + "-" * 60)
    print("Creating model...")
    model_args = ModelArgs(
        d_model=config['d_model'],
        n_layer=config['n_layer'],
        d_state=config['d_state'],
        expand=config['expand']
    )
    
    model = TrajectoryMambaModel(model_args, input_dim=config['input_dim']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Loss and optimizer
    criterion = nn.L1Loss()  # Changed from MSELoss to L1Loss
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint if specified
    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create log file for L1 losses
    log_file = checkpoint_dir / "training_log.txt"
    with open(log_file, 'w') as f:
        f.write("Epoch\tTrain_L1_Loss\tVal_L1_Loss\n")
    
    if not args.from_scratch and args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
        if checkpoint_path.exists():
            start_epoch, train_losses, val_losses = load_checkpoint(
                model, optimizer, str(checkpoint_path), device
            )
            best_val_loss = min(val_losses) if val_losses else float('inf')
    elif not args.from_scratch:
        # Try to load latest checkpoint
        import re
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            # Sort by epoch number (numeric) instead of alphabetically
            def get_epoch_number(path):
                match = re.search(r'checkpoint_epoch_(\d+)\.pt', path.name)
                return int(match.group(1)) if match else 0
            checkpoints.sort(key=get_epoch_number)
            latest = checkpoints[-1]
            print(f"\nFound checkpoint: {latest}")
            start_epoch, train_losses, val_losses = load_checkpoint(
                model, optimizer, str(latest), device
            )
            best_val_loss = min(val_losses) if val_losses else float('inf')
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Starting from epoch {start_epoch + 1}")
    print(f"Training for {config['num_epochs']} epochs")
    print(f"Batch size: {config['batch_size']}")
    
    end_epoch = start_epoch + config['num_epochs']
    
    for epoch in range(start_epoch, end_epoch):
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch + 1}/{end_epoch}")
        print(f"{'=' * 40}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, verbose=True
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nTrain Loss (L1): {train_loss:.6f}")
        print(f"Val Loss (L1): {val_loss:.6f}")
        print(f"LR: {current_lr:.2e}")
        
        # Write L1 losses to log file
        with open(log_file, 'a') as f:
            f.write(f"{epoch + 1}\t{train_loss:.6f}\t{val_loss:.6f}\n")
        
        # Save checkpoint every 5 epochs
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, str(checkpoint_path), config)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = checkpoint_dir / "best_model.pt"
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, str(best_path), config)
            print(f"  ★ New best model! Val Loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config['early_stopping_patience']})")
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n⚠ Early stopping at epoch {epoch + 1}")
            break
    
    # Save final checkpoint
    final_path = checkpoint_dir / "final_model.pt"
    save_checkpoint(model, optimizer, epoch, train_losses[-1], val_losses[-1], str(final_path), config)
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final epoch: {epoch + 1}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
