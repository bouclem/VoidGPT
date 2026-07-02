"""Training script for VoidGPT 1 120K."""

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from tokenizer import CharTokenizer
from dataset import CharDataset, create_datasets, load_text, encode_text
from model import VoidGPT120K


def get_lr(iteration: int, config: TrainConfig) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if iteration < config.warmup_iters:
        return config.learning_rate * iteration / config.warmup_iters
    if iteration > config.max_iters:
        return config.min_lr
    decay_ratio = (iteration - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def loss_to_ppl(loss: float) -> float:
    """Convert cross-entropy loss to perplexity."""
    return math.exp(min(loss, 20.0))


@torch.no_grad()
def evaluate(model: VoidGPT120K, loader: DataLoader, device: torch.device, eval_iters: int) -> tuple[float, float]:
    """Evaluate model on a data loader, return (avg_loss, perplexity)."""
    model.eval()
    total_loss = 0.0
    count = 0
    for x, y in loader:
        if count >= eval_iters:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    model.train()
    avg_loss = total_loss / max(count, 1)
    return avg_loss, loss_to_ppl(avg_loss)


def fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    if seconds < 0:
        return "--:--"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def train(
    model: VoidGPT120K,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_config: TrainConfig,
    device: torch.device,
    checkpoint_dir: Path,
):
    """Main training loop."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=train_config.betas,
        weight_decay=train_config.weight_decay,
        eps=1e-9,
    )

    best_val_loss = float("inf")
    train_iter = iter(train_loader)

    # Async checkpoint saving
    _save_thread = None

    def async_save_checkpoint(state_dict, path, extra_data):
        """Save checkpoint in a background thread to avoid blocking training.

        state_dict is captured by reference; the CPU copy happens inside the
        thread so the main training loop is not stalled by GPU->CPU transfer.
        """
        nonlocal _save_thread
        if _save_thread is not None:
            _save_thread.join()

        def _save():
            cpu_state = {k: v.cpu().clone() for k, v in state_dict.items()}
            data = {**extra_data, "model_state": cpu_state}
            torch.save(data, path)

        _save_thread = threading.Thread(target=_save, daemon=True)
        _save_thread.start()

    # Real-time tracking
    running_loss = 0.0
    running_loss_count = 0
    t_start = time.time()
    t_last = t_start
    tokens_seen = 0
    seq_len = train_config.batch_size * 128  # approx tokens per iter

    # Loss history for plotting
    history = {"iters": [], "train_loss": [], "val_loss": [], "val_ppl": []}

    model.train()

    print(f"\n{'='*70}")
    print(f"Training VoidGPT 1 120K — {train_config.max_iters} iterations")
    print(f"{'='*70}\n")

    for iteration in range(train_config.max_iters):
        lr = get_lr(iteration, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

        running_loss += loss.item()
        running_loss_count += 1
        tokens_seen += seq_len

        # Real-time display every 10 iterations
        if iteration % 10 == 0 or iteration == train_config.max_iters - 1:
            now = time.time()
            elapsed = now - t_start
            iter_time = now - t_last
            t_last = now

            avg_loss = running_loss / max(running_loss_count, 1)
            ppl = loss_to_ppl(avg_loss)
            progress = (iteration + 1) / train_config.max_iters * 100
            tps = tokens_seen / max(elapsed, 1e-6)
            eta = (train_config.max_iters - iteration - 1) * (elapsed / max(iteration + 1, 1))

            sys.stdout.write(
                f"\r  [{progress:5.1f}%] iter {iteration:5d}/{train_config.max_iters} "
                f"| loss {avg_loss:.4f} | ppl {ppl:7.2f} "
                f"| lr {lr:.2e} | {tps:,.0f} tok/s "
                f"| elapsed {fmt_time(elapsed)} | eta {fmt_time(eta)}   "
            )
            sys.stdout.flush()

            if iteration % 100 == 0 and iteration > 0:
                running_loss = 0.0
                running_loss_count = 0

        # Full eval with newline
        if iteration % train_config.eval_interval == 0 or iteration == train_config.max_iters - 1:
            sys.stdout.write("\r  evaluating...          ")
            sys.stdout.flush()
            val_loss, val_ppl = evaluate(model, val_loader, device, train_config.eval_iters)
            train_loss, train_ppl = evaluate(model, train_loader, device, train_config.eval_iters)

            sys.stdout.write("\n")
            sys.stdout.flush()
            print(
                f"  ┌─ EVAL iter {iteration}\n"
                f"  │  train: loss {train_loss:.4f} | ppl {train_ppl:7.2f}\n"
                f"  │  val:   loss {val_loss:.4f} | ppl {val_ppl:7.2f}\n"
                f"  └─ {'improved' if val_loss < best_val_loss else 'no improvement'}"
            )

            history["iters"].append(iteration)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_ppl"].append(val_ppl)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                async_save_checkpoint(
                    model.state_dict(),
                    checkpoint_dir / "best_model.pt",
                    {"config": model.config, "iter": iteration, "val_loss": val_loss, "val_ppl": val_ppl},
                )
                print(f"  ★ saved best model (val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f})")

        if iteration > 0 and iteration % train_config.checkpoint_interval == 0:
            async_save_checkpoint(
                model.state_dict(),
                checkpoint_dir / f"checkpoint_{iteration}.pt",
                {"config": model.config, "iter": iteration, "val_loss": val_loss, "val_ppl": val_ppl},
            )

    total_time = time.time() - t_start
    sys.stdout.write("\n")
    sys.stdout.flush()
    print(f"\n{'='*70}")
    print(f"Training complete in {fmt_time(total_time)}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best val ppl:  {loss_to_ppl(best_val_loss):.2f}")
    print(f"  Total tokens:  {tokens_seen:,}")
    print(f"  Avg speed:     {tokens_seen / max(total_time, 1e-6):,.0f} tok/s")
    print(f"{'='*70}")

    # Wait for any pending checkpoint save to finish
    if _save_thread is not None:
        _save_thread.join()

    # Save loss curve plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        ax.plot(history["iters"], history["train_loss"], label="train", alpha=0.8)
        ax.plot(history["iters"], history["val_loss"], label="val", alpha=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(history["iters"], history["val_ppl"], label="val PPL", color="steelblue", alpha=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Perplexity")
        ax.set_title("Validation Perplexity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = checkpoint_dir / "loss_curve.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Loss curve saved to {plot_path}")
    except ImportError:
        print("matplotlib not installed, skipping loss curve plot")

    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train VoidGPT 1 120K")
    parser.add_argument("--data", type=str, default="data/train.txt", help="Path to training text file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--max_iters", type=int, default=5000, help="Max training iterations")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--recursion_steps", type=int, default=1, help="Recursive weight reuse steps (1=standard)")
    parser.add_argument("--rmsnorm", action="store_true", help="Use RMSNorm instead of LayerNorm")
    parser.add_argument("--swiglu", action="store_true", help="Use SwiGLU FFN instead of GELU FFN")
    parser.add_argument("--rope", action="store_true", help="Use RoPE positional encoding instead of learned embeddings")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=256, help="FFN inner dimension")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data and build tokenizer
    text = load_text(args.data)
    tokenizer = CharTokenizer()
    tokenizer.build_from_text(text)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Save tokenizer
    tokenizer.save(Path(args.checkpoint_dir) / "tokenizer.json")

    # Create datasets
    train_ds, val_ds = create_datasets(args.data, tokenizer, seq_len=128, train_frac=0.9)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Create model
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        recursion_steps=args.recursion_steps,
        use_rmsnorm=args.rmsnorm,
        use_swiglu=args.swiglu,
        use_rope=args.rope,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
    )
    model = VoidGPT120K(model_config).to(device)

    param_counts = model.count_parameters()
    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    print(f"  Total: {param_counts['total']:,} ({param_counts['total'] / 1000:.1f}K)")

    # Train
    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

    best_val_loss = train(
        model, train_loader, val_loader, train_config, device, Path(args.checkpoint_dir)
    )

    print(f"\nFinal best validation loss: {best_val_loss:.4f}")
    print(f"Final perplexity: {loss_to_ppl(best_val_loss):.2f}")


if __name__ == "__main__":
    main()
