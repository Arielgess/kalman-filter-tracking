"""
Training script for Mamba model.
Loads checkpoint from epoch 5 and continues training.
Saves eval MSEs to a file and checkpoints after each epoch.

Usage:
    python train_mamba.py [--num_epochs N] [--checkpoint_path PATH]
    
Arguments:
    --num_epochs: Number of additional epochs to train (default: 10)
    --checkpoint_path: Path to checkpoint file (default: checkpoints/checkpoint_epoch_5.pt)
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add workspace root to path to import from src
script_dir = Path(__file__).parent
# Script is at: src/mamba/cv_training/train_mamba.py
# We need to add the workspace root (parent of src) to the path
workspace_root = script_dir.parent.parent.parent
sys.path.insert(0, str(workspace_root))

from src.mamba.mamba_model import TrajectoryMambaModel, ModelArgs
from src.mamba.trajectory_dataloader import TrajectoryDataset
from src.mamba.train_functions import (
    collate_fn, train_epoch, validate, save_checkpoint
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Check GPU details
if torch.cuda.is_available():
    print(f"\n✓ GPU is available!")
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Count: {torch.cuda.device_count()}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  Current GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
else:
    print(f"\n✗ GPU is NOT available - training will be slower on CPU")


# Configuration
config = {
    # Data
    'data_dir': str(script_dir / 'data' / 'data'),
    'train_split': 0.8,
    'val_split': 0.2,
    
    # Data Dimensions
    'input_dim': 2,        # Your actual data (x, y)
    'num_classes': 3,      # Your classes (e.g., Low Noise, High Noise, etc.)
    
    # Model "Brain" Dimensions
    'd_model': 64,        # The hidden size (Make this big! 64, 128, or 256)
    'n_layer': 4,          # Number of Mamba layers
    'd_state': 16,         # Standard default
    'expand': 2,           # Standard default
    
    # Training
    'batch_size': 32,      
    'learning_rate': 1e-3,
    'num_epochs': 10,  # Number of additional epochs to train after loading checkpoint
    'min_seq_length': 2,  # Minimum sequence length (sequences shorter than this will be filtered)
    'early_stopping_patience': 10,  # Early stopping patience (epochs without improvement)
    
    # Checkpointing
    'checkpoint_dir': str(script_dir / 'checkpoints'),
    'checkpoint_epoch_5_path': str(script_dir / 'checkpoints' / 'checkpoint_epoch_5.pt'),
    'eval_mse_file': str(script_dir / 'eval_mses.txt'),  # File to store eval MSEs
    
    # Other
    'num_workers': 0,
    'seed': 42
}

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Mamba model from checkpoint')
parser.add_argument('--num_epochs', type=int, default=None,
                    help='Number of additional epochs to train (default: from config)')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Path to checkpoint file (default: checkpoints/checkpoint_epoch_5.pt)')
args = parser.parse_args()

# Override config with command line arguments if provided
if args.num_epochs is not None:
    config['num_epochs'] = args.num_epochs
if args.checkpoint_path is not None:
    config['checkpoint_epoch_5_path'] = args.checkpoint_path

# Set random seed
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(config['seed'])

print("\nConfiguration:")
for key, value in config.items():
    print(f"  {key}: {value}")


# Load dataset
data_path = Path(config['data_dir'])
if not data_path.exists():
    raise ValueError(f"Data directory not found: {data_path}")

dataset = TrajectoryDataset(str(data_path))
print(f"\nTotal trajectories: {len(dataset)}")

# Split into train and validation
train_size = int(config['train_split'] * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset, 
    [train_size, val_size], 
    generator=torch.Generator().manual_seed(config['seed'])
)

print(f"Train trajectories: {len(train_dataset)}")
print(f"Validation trajectories: {len(val_dataset)}")

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['num_workers'],
    collate_fn=lambda batch: collate_fn(batch, min_seq_length=config['min_seq_length'])
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config['num_workers'],
    collate_fn=lambda batch: collate_fn(batch, min_seq_length=config['min_seq_length'])
)


# Create model
model_args = ModelArgs(
    d_model=config['d_model'],
    n_layer=config['n_layer'],
    d_state=config['d_state'],
    expand=config['expand']
)

model = TrajectoryMambaModel(model_args, input_dim=config["input_dim"]).to(device)
print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=config['learning_rate'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Create checkpoint directory
checkpoint_dir = Path(config['checkpoint_dir'])
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Initialize eval MSE file (create or append)
eval_mse_file = Path(config['eval_mse_file'])
if not eval_mse_file.exists():
    with open(eval_mse_file, 'w') as f:
        f.write("# Eval MSE per epoch\n")
    print(f"\nCreated eval MSE file: {eval_mse_file}")
else:
    print(f"\nEval MSE file exists: {eval_mse_file} (will append)")


# Load checkpoint from epoch 5
checkpoint_path = Path(config['checkpoint_epoch_5_path'])
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

print(f"\nLoading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load model state dict
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Get starting epoch (epoch 5 means we start from epoch 6, which is index 5)
start_epoch = checkpoint.get('epoch', 4) + 1  # Epoch 5 (0-indexed: 4) -> start from epoch 6 (0-indexed: 5)

# Get previous losses if available
train_losses = checkpoint.get('train_loss', [])
if not isinstance(train_losses, list):
    train_losses = [train_losses] if train_losses else []

val_losses = checkpoint.get('val_loss', [])
if not isinstance(val_losses, list):
    val_losses = [val_losses] if val_losses else []

print(f"✓ Model loaded successfully!")
print(f"  Starting from epoch: {start_epoch + 1} (0-indexed: {start_epoch})")
print(f"  Previous train losses: {train_losses}")
print(f"  Previous val losses: {val_losses}")

# Initialize best_val_loss from previous validation losses
best_val_loss = min(val_losses) if val_losses else float('inf')
patience_counter = 0

# Training history
end_epoch = start_epoch + config['num_epochs']
final_epoch = start_epoch  # Will be updated during training
print(f"\nStarting training from epoch {start_epoch + 1}...")
print(f"Total epochs to train: {config['num_epochs']}")
print(f"Will train until epoch {end_epoch} (0-indexed: {end_epoch - 1})")

for epoch in range(start_epoch, end_epoch):
    final_epoch = epoch
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{end_epoch}")
    print(f"{'='*50}")
    
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    
    # Validate
    val_loss = validate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Save eval MSE to file (append new line)
    with open(eval_mse_file, 'a') as f:
        f.write(f"{val_loss:.6f}\n")
    print(f"Eval MSE saved to {eval_mse_file}")
    
    # Save checkpoint after each epoch
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
    save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path, config)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_checkpoint_path = checkpoint_dir / "best_model.pt"
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, best_checkpoint_path, config)
        print(f"New best model saved! Val Loss: {val_loss:.6f}")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= config['early_stopping_patience']:
        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
        print(f"Best validation loss: {best_val_loss:.6f}")
        break

print(f"\n{'='*50}")
print("Training completed!")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Final epoch: {final_epoch + 1}")
print(f"Eval MSEs saved to: {eval_mse_file}")
print(f"{'='*50}")

