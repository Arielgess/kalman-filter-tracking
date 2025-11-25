import numpy as np
import torch
from tqdm import tqdm


def get_predictions(model, dataloader, device):
    model.eval()

    results = {
        "preds": [],
        "targets": [],
        "kf_params": [],  # List of dicts
        "real_positions": []
    }

    print("Generating predictions...")
    with torch.no_grad():
        for batch in dataloader:
            if batch is None: continue

            # 1. Load Data
            X = batch['X'].to(device)
            dt = batch['dt'].to(device)

            # Load your specific noise parameters directly from the batch
            # Assuming they are 1D tensors/arrays of shape (Batch_Size,)
            vel_stds = batch['vel_change_std']
            meas_stds = batch['measurement_noise_std']

            # Load un-normalization stats
            batch_mean = batch['X_mean'].to(device).unsqueeze(1)
            batch_std = batch['X_std'].to(device).unsqueeze(1)

            # 2. Run Model
            X_input = X[:, :-1, :]
            dt_input = dt[:, :-1].unsqueeze(-1)
            preds_norm = model(X_input, dt_input)

            # 3. Un-normalize
            preds_real = (preds_norm * batch_std) + batch_mean
            targets_real = (X[:, 1:, :] * batch_std) + batch_mean

            preds_np = preds_real.cpu().numpy()
            targets_np = targets_real.cpu().numpy()
            real_positions_np = batch['Y'].cpu().numpy()
            # 4. Loop over batch to package everything
            batch_size = preds_np.shape[0]

            for i in range(batch_size):
                # Save trajectories
                results["preds"].append(preds_np[i])
                results["targets"].append(targets_np[i])
                results["real_positions"].append(real_positions_np[i])

                # Construct the Dictionary for this specific trajectory
                # .item() converts 0-dim tensor or numpy scalar to a standard Python float
                kf_dict = {
                    "vel_change_std": vel_stds[i].item(),
                    "measurement_noise_std": meas_stds[i].item()
                }
                results["kf_params"].append(kf_dict)

    print(f"Generated {len(results['preds'])} trajectories.")
    return results

def collate_fn(batch, min_seq_length=2):
    """
    Custom collate function to handle variable-length sequences.
    Trims all sequences to the length of the shortest one in the batch.
    """
    # Filter out sequences that are too short
    print("Got to the collate")
    # Try to access config as global, otherwise use default

    batch = [item for item in batch if item['X'].shape[0] >= min_seq_length]

    if len(batch) == 0:
        return None

    # Get minimum length in batch (trim all to this length)
    min_len = min(item['X'].shape[0] for item in batch)
    # Trim sequences to min_len
    X_batch = []
    Y_batch = []
    dt_batch = []
    X_mean_batch = np.stack([item['X_mean'] for item in batch])
    X_std_batch = np.stack([item['X_std'] for item in batch])
    vel_change_std_batch = np.stack([item['vel_change_std'] for item in batch])
    measurement_noise_std_batch = np.stack([item['measurement_noise_std'] for item in batch])
    

    for item in batch:
        X = item['X']
        Y = item['Y'] if item['Y'] is not None else X  # Use X as target if Y not available
        dt = item['dt']

        # Trim to min_len (take first min_len timesteps)
        X = X[:min_len, :]
        Y = Y[:min_len, :]

        X_batch.append(X)
        Y_batch.append(Y)
        dt_batch.append(np.full(min_len, dt))  # Create dt array for trimmed sequence

    # Convert to tensors
    X_tensor = torch.FloatTensor(np.array(X_batch))  # (batch, min_len, 2)
    Y_tensor = torch.FloatTensor(np.array(Y_batch))  # (batch, min_len, 2)
    dt_tensor = torch.FloatTensor(np.array(dt_batch))  # (batch, min_len)
    lengths_tensor = torch.LongTensor([min_len] * len(batch))  # All same length

    return {
        'X': X_tensor,
        'Y': Y_tensor,
        'dt': dt_tensor,
        'lengths': lengths_tensor,
        'X_mean': torch.FloatTensor(X_mean_batch),
        'X_std': torch.FloatTensor(X_std_batch),
        'vel_change_std': torch.FloatTensor(vel_change_std_batch),
        'measurement_noise_std': torch.FloatTensor(measurement_noise_std_batch)
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    print("Got here")
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        print(f"About to start batch {num_batches + 1}")
        if batch is None:
            continue

        X = batch['X'].to(device)  # (batch, seq_len, 2)
        Y = batch['Y'].to(device)  # (batch, seq_len, 2)
        dt = batch['dt'].to(device)  # (batch, seq_len)
        lengths = batch['lengths']

        # Prepare input: use X[:-1] to predict X[1:]
        # This is a next-step prediction task
        seq_len = X.shape[1]
        if seq_len < 2:
            continue

        # Input: all but last timestep
        X_input = X[:, :-1, :]  # (batch, seq_len-1, 2)
        # Target: all but first timestep (next positions)
        Y_target = X[:, 1:, :]  # (batch, seq_len-1, 2)
        dt_input = dt[:, :-1]  # (batch, seq_len-1)

        # Expand dt to (batch, seq_len-1, 1) for model
        dt_input = dt_input.unsqueeze(-1)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_input, dt_input)  # (batch, seq_len-1, 2)

        # Compute loss (only on valid positions, mask out padding)
        loss = criterion(predictions, Y_target)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            print(f"About to start batch {num_batches + 1}")
            if batch is None:
                continue

            X = batch['X'].to(device)
            Y = batch['Y'].to(device)
            dt = batch['dt'].to(device)
            lengths = batch['lengths']

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


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path, config):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {start_epoch}")
    return start_epoch, checkpoint.get('train_loss', []), checkpoint.get('val_loss', [])
