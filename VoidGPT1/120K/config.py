"""VoidGPT 1 120K — Model and training configuration."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Transformer architecture configuration targeting ~120K parameters.

    Supports recursive weight reuse: when recursion_steps > 1, the transformer
    blocks are applied multiple times with shared weights, giving the model more
    'thinking depth' without adding parameters.
    """

    vocab_size: int = 100
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    max_seq_len: int = 128
    dropout: float = 0.1
    bias: bool = True
    weight_tying: bool = True
    recursion_steps: int = 1
    use_rmsnorm: bool = False
    use_swiglu: bool = False
    use_rope: bool = False
    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000


@dataclass
class GenerateConfig:
    """Text generation parameters."""

    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40
    seed: int = 42
