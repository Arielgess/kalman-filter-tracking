"""
Train Mamba using OLD dataloader and train functions.

This uses the original trajectory_dataloader.py approach (loading JSON files)
to test if it's faster than the tensor-based approach.
"""

import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, random_split

# Setup path
script_dir = Path(__file__).parent
workspace_root = script_dir.parent.parent.parent
sys.path.insert(0, str(workspace_root))

from src.mamba.mamba_model import TrajectoryMambaModel, ModelArgs
from src.mamba.trajectory_dataloader import TrajectoryDataset


def collate_fn_imm(batch, min_seq_length=2):
    """
    Custom collate function adapted for IMM data format.
    The original collate_fn expects class names like 'cv_0.5_vel_change_0.8_obs_noise'.
    This one handles the Universal_Physics IMM data.
    """
    import torch
    import numpy as np
    
    # Filter out sequences that are too short
    batch = [item for item in batch if item['X'].shape[0] >= min_seq_length]
    
    if len(batch) == 0:
        return None
    
    # Get minimum length in batch (trim all to this length)
    min_len = min(item['X'].shape[0] for item in batch)
    
    X_batch = []
    Y_batch = []
    dt_batch = []
    X_mean_batch = np.stack([item['X_mean'] for item in batch])
    X_std_batch = np.stack([item['X_std'] for item in batch])
    
    # For IMM data, use defaults since we don't have simple vel_change_std/measurement_noise_std
    # The original values from meta might be arrays or None
    measurement_noise_stds = []
    
    for item in batch:
        X = item['X']
        Y = item['Y'] if item['Y'] is not None else X
        dt = item['dt']
        
        # Trim to min_len
        X = X[:min_len, :]
        Y = Y[:min_len, :]
        
        X_batch.append(X)
        Y_batch.append(Y)
        dt_batch.append(np.full(min_len, dt))
        
        # Extract measurement_noise_std from meta if available
        meta = item.get('meta', {})
        if 'measurement_noise_std' in meta:
            mns = meta['measurement_noise_std']
            if isinstance(mns, (list, np.ndarray)):
                measurement_noise_stds.append(np.mean(mns))  # Average if 2D
            else:
                measurement_noise_stds.append(float(mns))
        else:
            measurement_noise_stds.append(0.3)  # Default
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(np.array(X_batch))
    Y_tensor = torch.FloatTensor(np.array(Y_batch))
    dt_tensor = torch.FloatTensor(np.array(dt_batch))
    
    return {
        'X': X_tensor,
        'Y': Y_tensor,
        'dt': dt_tensor,
        'lengths': torch.LongTensor([min_len] * len(batch)),
        'X_mean': torch.FloatTensor(X_mean_batch),
        'X_std': torch.FloatTensor(X_std_batch),
        'vel_change_std': torch.FloatTensor([0.3] * len(batch)),  # Placeholder
        'measurement_noise_std': torch.FloatTensor(measurement_noise_stds)
    }


def train_epoch_timed(model, dataloader, criterion, optimizer, device, verbose=True):
    """Train for one epoch with timing."""
    model.train()
    total_loss = 0
    num_batches = 0
    total_batches = len(dataloader)
    epoch_start = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue
        
        batch_start = time.time()
        
        X = batch['X'].to(device)
        dt = batch['dt'].to(device)
        
        seq_len = X.shape[1]
        if seq_len < 2:
            continue
        
        X_input = X[:, :-1, :]
        Y_target = X[:, 1:, :]
        dt_input = dt[:, :-1].unsqueeze(-1)
        
        optimizer.zero_grad()
        predictions = model(X_input, dt_input)
        loss = criterion(predictions, Y_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        batch_time = time.time() - batch_start
        elapsed = time.time() - epoch_start
        eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1) if batch_idx > 0 else 0
        print(f"  Batch {batch_idx + 1}/{total_batches} | Loss: {loss.item():.6f} | "
              f"Time: {batch_time:.2f}s | ETA: {eta/60:.1f}min")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_timed(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
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


def main():
    parser = argparse.ArgumentParser(description='Train Mamba with OLD dataloader')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()
    
    # Config
    config = {
        'data_dir': str(script_dir / 'data' / 'data'),
        'train_split': 0.9,
        'input_dim': 2,
        'd_model': 64,
        'n_layer': 4,
        'd_state': 16,
        'expand': 2,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'checkpoint_dir': str(script_dir / 'checkpoints_old'),
        'seed': 42
    }
    
    print("=" * 60)
    print("TRAINING WITH OLD DATALOADER (JSON-based)")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Load dataset
    print(f"\nLoading data from: {config['data_dir']}")
    dataset = TrajectoryDataset(config['data_dir'])
    print(f"Total trajectories: {len(dataset)}")
    
    # Check sample
    sample = dataset[0]
    print(f"Sample X shape: {sample['X'].shape}")
    print(f"Sequence length: {sample['X'].shape[0]}")
    
    # Split
    train_size = int(config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn_imm,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn_imm,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    model_args = ModelArgs(
        d_model=config['d_model'],
        n_layer=config['n_layer'],
        d_state=config['d_state'],
        expand=config['expand']
    )
    model = TrajectoryMambaModel(model_args, input_dim=config['input_dim']).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    
    # Create checkpoint dir
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")
        
        # Train
        train_loss = train_epoch_timed(model, train_loader, criterion, optimizer, device, verbose=True)
        
        # Validate
        val_loss = validate_timed(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Time:       {epoch_time/60:.1f} min")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = Path(config['checkpoint_dir']) / 'best_model.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  ✓ New best model saved!")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 60)


if __name__ == '__main__':
    main()

