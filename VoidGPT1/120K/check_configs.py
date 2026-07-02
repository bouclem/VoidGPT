"""Quick param count check for different model configs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ModelConfig
from model import VoidGPT120K

configs = [
    ("d64_l2_rec3 (current best)", ModelConfig(vocab_size=87, use_rope=True, use_swiglu=True, use_rmsnorm=True, recursion_steps=3)),
    ("d128_l1_rec3", ModelConfig(vocab_size=87, d_model=128, n_heads=4, n_layers=1, d_ff=512, use_rope=True, use_swiglu=True, use_rmsnorm=True, recursion_steps=3)),
    ("d128_l2_rec1", ModelConfig(vocab_size=87, d_model=128, n_heads=4, n_layers=2, d_ff=512, use_rope=True, use_swiglu=True, use_rmsnorm=True, recursion_steps=1)),
    ("d96_l2_rec2", ModelConfig(vocab_size=87, d_model=96, n_heads=4, n_layers=2, d_ff=384, use_rope=True, use_swiglu=True, use_rmsnorm=True, recursion_steps=2)),
    ("d96_l1_rec3", ModelConfig(vocab_size=87, d_model=96, n_heads=4, n_layers=1, d_ff=384, use_rope=True, use_swiglu=True, use_rmsnorm=True, recursion_steps=3)),
    ("d96_l1_rec3_dff256", ModelConfig(vocab_size=87, d_model=96, n_heads=4, n_layers=1, d_ff=256, use_rope=True, use_swiglu=True, use_rmsnorm=True, recursion_steps=3)),
    ("d80_l1_rec3", ModelConfig(vocab_size=87, d_model=80, n_heads=4, n_layers=1, d_ff=320, use_rope=True, use_swiglu=True, use_rmsnorm=True, recursion_steps=3)),
    ("d80_l2_rec2", ModelConfig(vocab_size=87, d_model=80, n_heads=4, n_layers=2, d_ff=320, use_rope=True, use_swiglu=True, use_rmsnorm=True, recursion_steps=2)),
]

for name, cfg in configs:
    model = VoidGPT120K(cfg)
    counts = model.count_parameters()
    print(f"{name:30s} | params {counts['total']:>7,} ({counts['total']/1000:.1f}K) | depth {counts['effective_depth']}")
