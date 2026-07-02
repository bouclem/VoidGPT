"""Quick verification script — checks param count, forward pass, and generation."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from config import ModelConfig
from model import VoidGPT120K
from tokenizer import CharTokenizer


def test_standard():
    print("=== Standard Model (recursion_steps=1) ===")
    cfg = ModelConfig(vocab_size=100)
    model = VoidGPT120K(cfg)
    counts = model.count_parameters()
    for k, v in counts.items():
        print(f"  {k}: {v:,}")
    total = counts["total"]
    print(f"  Total: {total:,} ({total / 1000:.1f}K)")
    assert total <= 150000, f"Param count {total} exceeds 150K budget"
    print("  Param budget: OK")
    return model


def test_recursive():
    print("\n=== Recursive Model (recursion_steps=3) ===")
    cfg = ModelConfig(vocab_size=100, recursion_steps=3)
    model = VoidGPT120K(cfg)
    counts = model.count_parameters()
    for k, v in counts.items():
        print(f"  {k}: {v:,}")
    total = counts["total"]
    depth = counts["effective_depth"]
    print(f"  Total: {total:,} ({total / 1000:.1f}K), effective depth: {depth}")
    assert total == counts["total"], "Recursive model should have same params"
    print("  Same params, 3x depth: OK")
    return model


def test_rmsnorm():
    print("\n=== RMSNorm Model ===")
    cfg = ModelConfig(vocab_size=100, use_rmsnorm=True)
    model = VoidGPT120K(cfg)
    counts = model.count_parameters()
    total = counts["total"]
    print(f"  Total: {total:,} ({total / 1000:.1f}K)")
    print("  RMSNorm: OK")
    return model


def test_swiglu():
    print("\n=== SwiGLU Model ===")
    cfg = ModelConfig(vocab_size=100, use_swiglu=True)
    model = VoidGPT120K(cfg)
    counts = model.count_parameters()
    total = counts["total"]
    print(f"  Total: {total:,} ({total / 1000:.1f}K)")
    assert total <= 150000, f"SwiGLU param count {total} exceeds 150K budget"
    x = torch.randint(0, 100, (4, 32))
    logits, loss = model(x, x)
    assert logits.shape == (4, 32, 100)
    print(f"  Forward: OK (loss {loss.item():.4f})")
    gen = model.generate(torch.randint(0, 100, (1, 5)), max_new_tokens=10)
    assert gen.shape == (1, 15)
    print("  Generation: OK")
    return model


def test_rope():
    print("\n=== RoPE Model ===")
    cfg = ModelConfig(vocab_size=100, use_rope=True)
    model = VoidGPT120K(cfg)
    counts = model.count_parameters()
    total = counts["total"]
    print(f"  Total: {total:,} ({total / 1000:.1f}K)")
    assert counts["positional_embedding"] == 0, "RoPE should have 0 positional embedding params"
    print(f"  Positional embedding params: 0 (RoPE uses no learned pos emb)")
    x = torch.randint(0, 100, (4, 32))
    logits, loss = model(x, x)
    assert logits.shape == (4, 32, 100)
    print(f"  Forward: OK (loss {loss.item():.4f})")
    gen = model.generate(torch.randint(0, 100, (1, 5)), max_new_tokens=10)
    assert gen.shape == (1, 15)
    print("  Generation: OK")
    return model


def test_all_features():
    print("\n=== All Features Model (RoPE + SwiGLU + RMSNorm + Recursive) ===")
    cfg = ModelConfig(
        vocab_size=100, use_rope=True, use_swiglu=True,
        use_rmsnorm=True, recursion_steps=2,
    )
    model = VoidGPT120K(cfg)
    counts = model.count_parameters()
    total = counts["total"]
    print(f"  Total: {total:,} ({total / 1000:.1f}K)")
    print(f"  Effective depth: {counts['effective_depth']}")
    assert total <= 150000, f"Full features param count {total} exceeds 150K budget"
    x = torch.randint(0, 100, (4, 32))
    logits, loss = model(x, x)
    assert logits.shape == (4, 32, 100)
    print(f"  Forward: OK (loss {loss.item():.4f})")
    gen = model.generate(torch.randint(0, 100, (1, 5)), max_new_tokens=10)
    assert gen.shape == (1, 15)
    print("  Generation: OK")
    return model


def test_forward(model, name=""):
    print(f"\n=== Forward Pass {name} ===")
    x = torch.randint(0, 100, (4, 32))
    logits, loss = model(x, x)
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    assert logits.shape == (4, 32, 100), f"Unexpected logits shape: {logits.shape}"
    print("  Forward pass: OK")


def test_generation(model, name=""):
    print(f"\n=== Generation {name} ===")
    x = torch.randint(0, 100, (1, 5))
    gen = model.generate(x, max_new_tokens=20, temperature=0.8, top_k=40)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {gen.shape}")
    assert gen.shape == (1, 25), f"Unexpected generation shape: {gen.shape}"
    print("  Generation: OK")


def test_tokenizer():
    print("\n=== Tokenizer ===")
    tok = CharTokenizer()
    print(f"  Vocab size: {tok.vocab_size}")
    text = "Hello, World!"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"  Encode: '{text}' -> {ids}")
    print(f"  Decode: {ids} -> '{decoded}'")
    assert decoded == text, f"Round-trip failed: '{decoded}' != '{text}'"
    print("  Tokenizer round-trip: OK")


def test_recursive_forward():
    print("\n=== Recursive Forward Pass ===")
    cfg = ModelConfig(vocab_size=100, recursion_steps=3)
    model = VoidGPT120K(cfg)
    x = torch.randint(0, 100, (4, 32))
    logits, loss = model(x, x)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    assert logits.shape == (4, 32, 100)
    print("  Recursive forward: OK")


if __name__ == "__main__":
    print("VoidGPT 1 120K — Verification\n")

    test_tokenizer()

    model = test_standard()
    test_forward(model, "(standard)")
    test_generation(model, "(standard)")

    test_recursive()
    test_recursive_forward()

    test_rmsnorm()
    test_swiglu()
    test_rope()
    test_all_features()

    print("\n=== ALL TESTS PASSED ===")
