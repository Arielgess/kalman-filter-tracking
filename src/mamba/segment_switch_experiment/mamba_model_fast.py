"""
Fast Mamba model using official mamba-ssm package with CUDA kernels.

Install: pip install mamba-ssm

This provides 10-20x speedup over the pure PyTorch einsum implementation.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

# Try to import official mamba-ssm
try:
    from mamba_ssm import Mamba as MambaSSM
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    print("⚠️  mamba-ssm not installed. Install with: pip install mamba-ssm")


@dataclass
class ModelArgs:
    """Model configuration."""
    d_model: int = 64
    n_layer: int = 4
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class FastMambaBlock(nn.Module):
    """Single Mamba block using official CUDA kernels."""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm = RMSNorm(args.d_model)
        
        # Use official Mamba with CUDA kernels
        self.mamba = MambaSSM(
            d_model=args.d_model,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
        )
    
    def forward(self, x):
        # Residual connection with pre-norm
        return self.mamba(self.norm(x)) + x


class FastMamba(nn.Module):
    """Fast Mamba backbone using official CUDA kernels."""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList([FastMambaBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
    
    def forward(self, x):
        """
        Args:
            x: shape (batch, seq_len, d_model)
        Returns:
            output: shape (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)


class TrajectoryMambaModelFast(nn.Module):
    """
    Fast trajectory prediction model using official Mamba CUDA kernels.
    
    This is a drop-in replacement for TrajectoryMambaModel with ~10-20x faster
    backward passes.
    
    Note: This version does NOT use the dt (time step) input since the official
    Mamba implementation doesn't support variable time steps. The model learns
    to predict with a fixed implicit time step.
    """
    
    def __init__(self, args: ModelArgs, input_dim: int = 2):
        super().__init__()
        
        if not MAMBA_SSM_AVAILABLE:
            raise ImportError(
                "mamba-ssm package not installed. "
                "Install with: pip install mamba-ssm"
            )
        
        self.args = args
        self.input_dim = input_dim
        
        # Encoder: 2D -> d_model
        self.encoder = nn.Linear(input_dim, args.d_model)
        
        # Backbone: Fast Mamba with CUDA kernels
        self.backbone = FastMamba(args)
        
        # Prediction head: d_model -> 2D
        self.pred_head = nn.Linear(args.d_model, input_dim)
    
    def forward(self, x, dt=None):
        """
        Args:
            x: Input trajectories (batch, seq_len, input_dim)
            dt: Time step - IGNORED in this fast version
                (kept for API compatibility)
        
        Returns:
            predictions: (batch, seq_len, input_dim)
        """
        # Encode to high-dimensional space
        x_features = self.encoder(x)  # (B, L, d_model)
        
        # Run through fast Mamba backbone
        hidden_states = self.backbone(x_features)  # (B, L, d_model)
        
        # Predict next position
        predictions = self.pred_head(hidden_states)  # (B, L, input_dim)
        
        return predictions


# Alias for easy import
TrajectoryMambaModel = TrajectoryMambaModelFast


def check_mamba_ssm():
    """Check if mamba-ssm is properly installed."""
    if not MAMBA_SSM_AVAILABLE:
        print("❌ mamba-ssm NOT installed")
        print("   Install with: pip install mamba-ssm")
        return False
    
    print("✓ mamba-ssm is installed")
    
    # Quick test
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args = ModelArgs(d_model=64, n_layer=2, d_state=16, expand=2)
        model = TrajectoryMambaModelFast(args, input_dim=2).to(device)
        
        x = torch.randn(2, 100, 2).to(device)
        with torch.no_grad():
            out = model(x)
        
        print(f"✓ Quick test passed: input {x.shape} -> output {out.shape}")
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


if __name__ == '__main__':
    check_mamba_ssm()

