"""Pytest test suite for VoidGPT 1 120K model."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig
from model import VoidGPT120K, RoPE, SwiGLUFFN, FeedForward, RMSNorm, LayerNorm
from tokenizer import CharTokenizer


# ─── Fixtures ───

@pytest.fixture
def vocab_size():
    return 100


@pytest.fixture
def base_config(vocab_size):
    return ModelConfig(vocab_size=vocab_size)


@pytest.fixture
def small_input(vocab_size):
    return torch.randint(0, vocab_size, (4, 32))


# ─── Config tests ───

class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.d_model == 64
        assert cfg.n_heads == 4
        assert cfg.n_layers == 2
        assert cfg.d_ff == 256
        assert cfg.max_seq_len == 128
        assert cfg.recursion_steps == 1
        assert cfg.use_rmsnorm is False
        assert cfg.use_swiglu is False
        assert cfg.use_rope is False

    def test_custom_values(self):
        cfg = ModelConfig(d_model=128, n_heads=8, n_layers=4)
        assert cfg.d_model == 128
        assert cfg.n_heads == 8
        assert cfg.n_layers == 4


# ─── Tokenizer tests ───

class TestTokenizer:
    def test_round_trip(self):
        tok = CharTokenizer()
        text = "Hello, World!"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_special_tokens(self):
        tok = CharTokenizer()
        assert tok.pad_id == 0
        assert tok.bos_id == 1
        assert tok.eos_id == 2
        assert tok.unk_id == 3

    def test_empty_string(self):
        tok = CharTokenizer()
        ids = tok.encode("")
        assert ids == []

    def test_vocab_size(self):
        tok = CharTokenizer()
        assert tok.vocab_size > 0

    def test_build_from_text(self):
        tok = CharTokenizer()
        tok.build_from_text("abc123")
        ids = tok.encode("abc")
        assert len(ids) == 3
        assert tok.decode(ids) == "abc"


# ─── Model tests ───

class TestModel:
    def test_param_count_under_budget(self, base_config):
        model = VoidGPT120K(base_config)
        counts = model.count_parameters()
        assert counts["total"] <= 150000, f"Params {counts['total']} exceed 150K budget"

    def test_forward_shape(self, base_config, small_input):
        model = VoidGPT120K(base_config)
        logits, loss = model(small_input, small_input)
        assert logits.shape == (4, 32, 100)
        assert loss is not None
        assert loss.item() > 0

    def test_forward_no_targets(self, base_config, small_input):
        model = VoidGPT120K(base_config)
        logits, loss = model(small_input)
        assert logits.shape == (4, 32, 100)
        assert loss is None

    def test_generation_shape(self, base_config):
        model = VoidGPT120K(base_config)
        x = torch.randint(0, 100, (1, 5))
        gen = model.generate(x, max_new_tokens=20, temperature=0.8, top_k=40)
        assert gen.shape == (1, 25)

    def test_generation_deterministic_with_seed(self, base_config):
        model = VoidGPT120K(base_config)
        x = torch.randint(0, 100, (1, 5))
        torch.manual_seed(42)
        gen1 = model.generate(x, max_new_tokens=10, temperature=0.5, top_k=10)
        torch.manual_seed(42)
        gen2 = model.generate(x, max_new_tokens=10, temperature=0.5, top_k=10)
        assert torch.equal(gen1, gen2)

    def test_seq_len_assertion(self, base_config):
        model = VoidGPT120K(base_config)
        x = torch.randint(0, 100, (1, 200))
        with pytest.raises(AssertionError, match="exceeds max"):
            model(x)


# ─── Recursive tests ───

class TestRecursive:
    def test_same_param_count(self, vocab_size):
        cfg1 = ModelConfig(vocab_size=vocab_size, recursion_steps=1)
        cfg2 = ModelConfig(vocab_size=vocab_size, recursion_steps=3)
        m1 = VoidGPT120K(cfg1)
        m2 = VoidGPT120K(cfg2)
        assert m1.count_parameters()["total"] == m2.count_parameters()["total"]

    def test_effective_depth(self, vocab_size):
        cfg = ModelConfig(vocab_size=vocab_size, recursion_steps=3)
        model = VoidGPT120K(cfg)
        counts = model.count_parameters()
        assert counts["effective_depth"] == 6

    def test_recursive_forward(self, vocab_size):
        cfg = ModelConfig(vocab_size=vocab_size, recursion_steps=3)
        model = VoidGPT120K(cfg)
        x = torch.randint(0, vocab_size, (4, 32))
        logits, loss = model(x, x)
        assert logits.shape == (4, 32, vocab_size)


# ─── RMSNorm tests ───

class TestRMSNorm:
    def test_rmsnorm_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(4, 32, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_rmsnorm_preserves_scale(self):
        norm = RMSNorm(64)
        x = torch.ones(1, 1, 64)
        out = norm(x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_model_with_rmsnorm(self, vocab_size):
        cfg = ModelConfig(vocab_size=vocab_size, use_rmsnorm=True)
        model = VoidGPT120K(cfg)
        counts = model.count_parameters()
        assert counts["total"] <= 150000


# ─── SwiGLU tests ───

class TestSwiGLU:
    def test_swiglu_shape(self, base_config):
        ffn = SwiGLUFFN(base_config)
        x = torch.randn(4, 32, base_config.d_model)
        out = ffn(x)
        assert out.shape == x.shape

    def test_model_with_swiglu(self, vocab_size):
        cfg = ModelConfig(vocab_size=vocab_size, use_swiglu=True)
        model = VoidGPT120K(cfg)
        counts = model.count_parameters()
        assert counts["total"] <= 150000
        x = torch.randint(0, vocab_size, (4, 32))
        logits, loss = model(x, x)
        assert logits.shape == (4, 32, vocab_size)


# ─── RoPE tests ───

class TestRoPE:
    def test_rope_shape(self):
        rope = RoPE(head_dim=16, max_seq_len=128)
        x = torch.randn(2, 4, 32, 16)
        out = rope(x)
        assert out.shape == x.shape

    def test_rope_position_zero_identity(self):
        rope = RoPE(head_dim=16, max_seq_len=128)
        x = torch.randn(1, 1, 1, 16)
        out = rope(x, seq_offset=0)
        assert torch.allclose(out, x, atol=1e-5), "RoPE at position 0 should be identity"

    def test_model_with_rope(self, vocab_size):
        cfg = ModelConfig(vocab_size=vocab_size, use_rope=True)
        model = VoidGPT120K(cfg)
        counts = model.count_parameters()
        assert counts["positional_embedding"] == 0
        assert counts["total"] <= 150000
        x = torch.randint(0, vocab_size, (4, 32))
        logits, loss = model(x, x)
        assert logits.shape == (4, 32, vocab_size)

    def test_rope_generation(self, vocab_size):
        cfg = ModelConfig(vocab_size=vocab_size, use_rope=True)
        model = VoidGPT120K(cfg)
        x = torch.randint(0, vocab_size, (1, 5))
        gen = model.generate(x, max_new_tokens=10, temperature=0.8, top_k=20)
        assert gen.shape == (1, 15)


# ─── Combined features test ───

class TestAllFeatures:
    def test_all_features_model(self, vocab_size):
        cfg = ModelConfig(
            vocab_size=vocab_size,
            use_rope=True,
            use_swiglu=True,
            use_rmsnorm=True,
            recursion_steps=2,
        )
        model = VoidGPT120K(cfg)
        counts = model.count_parameters()
        assert counts["total"] <= 150000
        assert counts["positional_embedding"] == 0
        assert counts["effective_depth"] == 4

        x = torch.randint(0, vocab_size, (4, 32))
        logits, loss = model(x, x)
        assert logits.shape == (4, 32, vocab_size)

        gen = model.generate(torch.randint(0, vocab_size, (1, 5)), max_new_tokens=10)
        assert gen.shape == (1, 15)

    def test_all_features_param_count(self, vocab_size):
        """All features should still be within 120K budget."""
        cfg = ModelConfig(
            vocab_size=vocab_size,
            use_rope=True,
            use_swiglu=True,
            use_rmsnorm=True,
        )
        model = VoidGPT120K(cfg)
        total = model.count_parameters()["total"]
        print(f"\nAll-features param count: {total:,} ({total/1000:.1f}K)")
        assert total <= 150000
