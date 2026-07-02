"""VoidGPT 1 120K — Decoder-only transformer model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    """RMSNorm — simpler than LayerNorm, fewer params (no bias, no mean subtraction)."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.layer_norm(x, x.shape[-1:], self.weight, self.bias, eps=1e-5)
        return out


def make_norm(dim: int, bias: bool, use_rms: bool) -> nn.Module:
    """Create normalization layer: RMSNorm or LayerNorm."""
    if use_rms:
        return RMSNorm(dim)
    return LayerNorm(dim, bias=bias)


class RoPE(nn.Module):
    """Rotary Position Embedding.

    Rotates Q and K pairs by angle m*theta_i where theta_i = theta^(-2i/d).
    Removes need for learned positional embeddings.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        # Precompute frequencies: theta_i = theta^(-2i/d) for i in [0, d/2)
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # Angles: m * theta_i for positions m in [0, max_seq_len)
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, freqs)  # (max_seq_len, head_dim/2)
        # Duplicate to match head_dim: cos and sin for each pair
        self.register_buffer("cos", torch.cos(angles).repeat(1, 2), persistent=False)
        self.register_buffer("sin", torch.sin(angles).repeat(1, 2), persistent=False)

    def forward(self, x: torch.Tensor, seq_offset: int = 0) -> torch.Tensor:
        """Apply rotary embedding to x of shape (B, n_heads, T, head_dim)."""
        T = x.shape[2]
        cos = self.cos[seq_offset:seq_offset + T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        sin = self.sin[seq_offset:seq_offset + T].unsqueeze(0).unsqueeze(0)
        # Rotate: x_rot = x * cos + rotate_half(x) * sin
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional RoPE."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = config.use_rope

        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        if config.use_rope:
            self.rope = RoPE(self.head_dim, config.max_seq_len, config.rope_theta)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (not V)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * self.scale
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network: (SiLU(xW1) * xW2) W3.

    Uses 3 linear layers. To match param count of standard FFN (2 layers),
    inner dim is (2/3) * d_ff. Used in LLaMA, PaLM, Mistral.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # (2/3) factor to keep param count comparable to standard FFN
        inner = int(config.d_ff * 2 / 3)
        # Ensure inner is even for clean split
        inner = inner + (inner % 2)
        self.w1 = nn.Linear(config.d_model, inner, bias=config.bias)
        self.w2 = nn.Linear(config.d_model, inner, bias=config.bias)
        self.w3 = nn.Linear(inner, config.d_model, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        b = F.silu(self.w2(x))
        return self.drop(self.w3(a * b))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: Norm → Attention → Residual, Norm → FFN → Residual."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = make_norm(config.d_model, config.bias, config.use_rmsnorm)
        self.attn = CausalSelfAttention(config)
        self.ln2 = make_norm(config.d_model, config.bias, config.use_rmsnorm)
        if config.use_swiglu:
            self.ffn = SwiGLUFFN(config)
        else:
            self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class VoidGPT120K(nn.Module):
    """VoidGPT 1 120K — decoder-only transformer with optional recursive weight reuse.

    Architecture:
        Token Embedding + Positional Embedding
        → Transformer Block × n_layers (repeated recursion_steps times)
        → Final LayerNorm
        → Output Projection (weight-tied with token embedding)

    When recursion_steps > 1, the same n_layers blocks are applied multiple
    times, giving the model more depth without adding parameters.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        # Only use learned positional embeddings if not using RoPE
        if config.use_rope:
            self.pos_emb = None
        else:
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = make_norm(config.d_model, config.bias, config.use_rmsnorm)

        # Output head — optionally weight-tied with token embedding
        if config.weight_tying:
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight  # tie weights
        else:
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=config.bias)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Forward pass.

        Args:
            idx: (B, T) token IDs
            targets: (B, T) target token IDs for loss computation

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar (if targets provided)
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        tok = self.tok_emb(idx)  # (B, T, d_model)
        if self.pos_emb is not None:
            pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, d_model)
            x = self.drop(tok + pos.unsqueeze(0))
        else:
            # RoPE handles positions inside attention; no positional embedding needed
            x = self.drop(tok)

        # Apply blocks recursively: same blocks reused recursion_steps times
        for _ in range(self.config.recursion_steps):
            for block in self.blocks:
                x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=0,  # ignore PAD tokens
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive text generation.

        Args:
            idx: (B, T) starting token IDs
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (lower = more deterministic)
            top_k: if set, only sample from top-k tokens

        Returns:
            (B, T + max_new_tokens) generated token IDs
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts["token_embedding"] = sum(p.numel() for p in self.tok_emb.parameters())
        counts["positional_embedding"] = sum(p.numel() for p in self.pos_emb.parameters()) if self.pos_emb is not None else 0
        counts["transformer_blocks"] = sum(p.numel() for p in self.blocks.parameters())
        counts["final_norm"] = sum(p.numel() for p in self.ln_f.parameters())
        if self.config.weight_tying:
            counts["output_head"] = 0  # tied, already counted in token_embedding
        else:
            counts["output_head"] = sum(p.numel() for p in self.head.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["effective_depth"] = self.config.n_layers * self.config.recursion_steps
        return counts
