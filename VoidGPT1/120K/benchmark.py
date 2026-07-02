"""Benchmark script — compare architecture variants of VoidGPT 1 120K.

Runs short training for each variant and plots loss curves.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from config import ModelConfig, TrainConfig
from tokenizer import CharTokenizer
from dataset import create_datasets, load_text
from model import VoidGPT120K
from train import get_lr, evaluate, loss_to_ppl, fmt_time


VARIANTS = {
    "d64_l2_rec3": {"d_model": 64, "n_heads": 4, "n_layers": 2, "d_ff": 256, "use_rope": True, "use_swiglu": True, "use_rmsnorm": True, "recursion_steps": 3},
    "d96_l1_rec3": {"d_model": 96, "n_heads": 4, "n_layers": 1, "d_ff": 384, "use_rope": True, "use_swiglu": True, "use_rmsnorm": True, "recursion_steps": 3},
    "d96_l1_rec3_dff256": {"d_model": 96, "n_heads": 4, "n_layers": 1, "d_ff": 256, "use_rope": True, "use_swiglu": True, "use_rmsnorm": True, "recursion_steps": 3},
    "d80_l1_rec3": {"d_model": 80, "n_heads": 4, "n_layers": 1, "d_ff": 320, "use_rope": True, "use_swiglu": True, "use_rmsnorm": True, "recursion_steps": 3},
    "d64_l2_rec2": {"d_model": 64, "n_heads": 4, "n_layers": 2, "d_ff": 256, "use_rope": True, "use_swiglu": True, "use_rmsnorm": True, "recursion_steps": 2},
    "d64_l2_rec1": {"d_model": 64, "n_heads": 4, "n_layers": 2, "d_ff": 256, "use_rope": True, "use_swiglu": True, "use_rmsnorm": True, "recursion_steps": 1},
}


def train_variant(
    name: str,
    model_kwargs: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
    max_iters: int,
    device: torch.device,
    eval_interval: int,
) -> dict:
    """Train a single variant and return metrics."""
    cfg = ModelConfig(vocab_size=vocab_size, **model_kwargs)
    model = VoidGPT120K(cfg).to(device)
    param_count = model.count_parameters()["total"]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9
    )

    train_config = TrainConfig(max_iters=max_iters, eval_interval=eval_interval)
    train_iter = iter(train_loader)
    history = {"iters": [], "train_loss": [], "val_loss": [], "val_ppl": []}

    model.train()
    t0 = time.time()

    for iteration in range(max_iters):
        lr = get_lr(iteration, train_config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if iteration % eval_interval == 0 or iteration == max_iters - 1:
            val_loss, val_ppl = evaluate(model, val_loader, device, 50)
            train_loss, _ = evaluate(model, train_loader, device, 50)
            history["iters"].append(iteration)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_ppl"].append(val_ppl)

    elapsed = time.time() - t0
    final_val_loss = history["val_loss"][-1]
    final_val_ppl = history["val_ppl"][-1]

    print(
        f"  {name:25s} | params {param_count:>7,} | "
        f"val_loss {final_val_loss:.4f} | val_ppl {final_val_ppl:7.2f} | "
        f"time {fmt_time(elapsed)}"
    )

    return {
        "name": name,
        "params": param_count,
        "val_loss": final_val_loss,
        "val_ppl": final_val_ppl,
        "time": elapsed,
        "history": history,
    }


def plot_results(results: list[dict], output_path: Path):
    """Plot loss curves for all variants."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Validation loss curves
    ax = axes[0]
    for r in results:
        h = r["history"]
        ax.plot(h["iters"], h["val_loss"], label=r["name"], alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 2: Perplexity curves
    ax = axes[1]
    for r in results:
        h = r["history"]
        ax.plot(h["iters"], h["val_ppl"], label=r["name"], alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation Perplexity")
    ax.set_title("Validation Perplexity")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 3: Final comparison bar chart
    ax = axes[2]
    names = [r["name"] for r in results]
    ppls = [r["val_ppl"] for r in results]
    params = [r["params"] / 1000 for r in results]
    x = range(len(names))
    bars = ax.bar(x, ppls, color="steelblue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Final Val PPL")
    ax.set_title("Final Perplexity Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{p:.0f}K", ha="center", fontsize=6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark VoidGPT 1 120K architecture variants")
    parser.add_argument("--data", type=str, default="data/train.txt")
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument("--plot", type=str, default="benchmark_plot.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Benchmarking {len(VARIANTS)} variants, {args.max_iters} iters each\n")

    # Load data
    text = load_text(args.data)
    tokenizer = CharTokenizer()
    tokenizer.build_from_text(text)
    vocab_size = tokenizer.vocab_size

    train_ds, val_ds = create_datasets(args.data, tokenizer, seq_len=128, train_frac=0.9)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print(f"Vocab size: {vocab_size}, Train samples: {len(train_ds)}\n")
    print(f"{'Variant':25s} | {'Params':>9s} | {'Val Loss':>9s} | {'Val PPL':>9s} | {'Time':>8s}")
    print("-" * 75)

    results = []
    for name, kwargs in VARIANTS.items():
        result = train_variant(
            name, kwargs, train_loader, val_loader, vocab_size,
            args.max_iters, device, args.eval_interval,
        )
        results.append(result)

    # Save results
    output_path = Path(args.output)
    serializable = []
    for r in results:
        serializable.append({k: v for k, v in r.items() if k != "history"})
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Plot
    plot_results(results, Path(args.plot))

    # Summary
    best = min(results, key=lambda r: r["val_ppl"])
    print(f"\n{'='*75}")
    print(f"Best variant: {best['name']} (val_ppl={best['val_ppl']:.2f}, params={best['params']:,})")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
