"""
Quick diagnostic to find the bottleneck.
"""
import sys
import time
from pathlib import Path

# Add workspace root to path
script_dir = Path(__file__).parent
workspace_root = script_dir.parent.parent.parent
sys.path.insert(0, str(workspace_root))

import torch
import torch.nn as nn

print("=" * 60)
print("SPEED DIAGNOSTIC")
print("=" * 60)

# 1. Check CUDA
print("\n1. CUDA CHECK:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    device = torch.device('cuda')
else:
    print("   ⚠️  NO CUDA - Running on CPU (this is the problem!)")
    device = torch.device('cpu')

# 2. Test tensor operations speed
print("\n2. TENSOR OPERATION SPEED:")
x = torch.randn(16, 500, 64).to(device)
y = torch.randn(16, 500, 64).to(device)

# Warm up
for _ in range(3):
    z = torch.matmul(x, y.transpose(-1, -2))
torch.cuda.synchronize() if torch.cuda.is_available() else None

start = time.time()
for _ in range(10):
    z = torch.matmul(x, y.transpose(-1, -2))
torch.cuda.synchronize() if torch.cuda.is_available() else None
elapsed = time.time() - start
print(f"   10x matmul (16, 500, 64): {elapsed:.3f}s ({elapsed/10*1000:.1f}ms each)")
if elapsed > 1.0:
    print("   ⚠️  SLOW - GPU might not be working properly")

# 3. Test model forward pass
print("\n3. MODEL FORWARD PASS:")
from src.mamba.mamba_model import TrajectoryMambaModel, ModelArgs

model_args = ModelArgs(d_model=64, n_layer=4, d_state=16, expand=2)
model = TrajectoryMambaModel(model_args, input_dim=2).to(device)
print(f"   Model on device: {next(model.parameters()).device}")

x_input = torch.randn(16, 499, 2).to(device)
dt_input = torch.full((16, 499, 1), 0.04).to(device)

# Warm up
model.eval()
with torch.no_grad():
    for _ in range(2):
        _ = model(x_input, dt_input)
torch.cuda.synchronize() if torch.cuda.is_available() else None

# Time forward pass
start = time.time()
with torch.no_grad():
    for _ in range(5):
        out = model(x_input, dt_input)
torch.cuda.synchronize() if torch.cuda.is_available() else None
elapsed = time.time() - start
print(f"   5x forward pass: {elapsed:.3f}s ({elapsed/5*1000:.1f}ms each)")

if elapsed/5 > 2.0:
    print("   ⚠️  VERY SLOW forward pass!")
elif elapsed/5 > 0.5:
    print("   ⚠️  Slower than expected")
else:
    print("   ✓ Forward pass speed OK")

# 4. Test data loading
print("\n4. DATA LOADING:")
tensor_path = script_dir / "data" / "tensor_dataset.pt"
if tensor_path.exists():
    start = time.time()
    data = torch.load(tensor_path)
    elapsed = time.time() - start
    print(f"   Load tensor file: {elapsed:.2f}s")
    print(f"   X shape: {data['X'].shape}")
    
    # Test dataloader
    from src.mamba.segment_switch_experiment.tensor_dataloader import (
        TensorTrajectoryDataset, tensor_collate_fn
    )
    from torch.utils.data import DataLoader, Subset
    
    dataset = TensorTrajectoryDataset(str(tensor_path), normalize=True, stats=None)
    subset = Subset(dataset, list(range(160)))  # 10 batches worth
    loader = DataLoader(subset, batch_size=16, shuffle=False, 
                       collate_fn=tensor_collate_fn, num_workers=0)
    
    start = time.time()
    for i, batch in enumerate(loader):
        X = batch['X'].to(device)
        dt = batch['dt'].to(device)
    elapsed = time.time() - start
    print(f"   Load 10 batches to GPU: {elapsed:.3f}s ({elapsed/10*1000:.1f}ms each)")
else:
    print(f"   ⚠️  Tensor file not found: {tensor_path}")

# 5. Full training step timing
print("\n5. FULL TRAINING STEP:")
model.train()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

x_input = torch.randn(16, 499, 2).to(device)
dt_input = torch.full((16, 499, 1), 0.04).to(device)
y_target = torch.randn(16, 499, 2).to(device)

# Warm up
optimizer.zero_grad()
out = model(x_input, dt_input)
loss = criterion(out, y_target)
loss.backward()
optimizer.step()
torch.cuda.synchronize() if torch.cuda.is_available() else None

# Time
start = time.time()
for _ in range(5):
    optimizer.zero_grad()
    out = model(x_input, dt_input)
    loss = criterion(out, y_target)
    loss.backward()
    optimizer.step()
torch.cuda.synchronize() if torch.cuda.is_available() else None
elapsed = time.time() - start
print(f"   5x training steps: {elapsed:.3f}s ({elapsed/5*1000:.1f}ms each)")

if elapsed/5 > 5.0:
    print("\n   ❌ PROBLEM: Training step takes >5s")
    print("      This should take <500ms on a decent GPU")
elif elapsed/5 > 1.0:
    print("\n   ⚠️  WARNING: Training step takes >1s (slow but usable)")
else:
    print("\n   ✓ Training step speed OK")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)

