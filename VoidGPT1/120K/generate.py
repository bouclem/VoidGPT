"""Text generation for VoidGPT 1 120K."""

import argparse
from pathlib import Path

import torch

from config import ModelConfig, GenerateConfig
from tokenizer import CharTokenizer
from model import VoidGPT120K


def generate_text(
    model: VoidGPT120K,
    tokenizer: CharTokenizer,
    prompt: str,
    config: GenerateConfig,
    device: torch.device,
) -> str:
    """Generate text from a prompt."""
    torch.manual_seed(config.seed)
    model.eval()

    ids = tokenizer.encode(prompt)
    if len(ids) == 0:
        ids = [tokenizer.bos_id]

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    generated = model.generate(
        idx,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
    )

    generated_ids = generated[0].tolist()
    text = tokenizer.decode(generated_ids)
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate text with VoidGPT 1 120K")
    parser.add_argument("--prompt", type=str, default="The Sun is", help="Text prompt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json", help="Tokenizer file")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = CharTokenizer.load(args.tokenizer)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = checkpoint["config"]
    model = VoidGPT120K(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded model from {args.checkpoint} (iter {checkpoint['iter']}, val_loss={checkpoint['val_loss']:.4f})")

    gen_config = GenerateConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )

    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            prompt = input("Prompt> ")
            if prompt.strip().lower() in ("quit", "exit", "q"):
                break
            text = generate_text(model, tokenizer, prompt, gen_config, device)
            print(f"\n{text}\n")
    else:
        text = generate_text(model, tokenizer, args.prompt, gen_config, device)
        print(text)


if __name__ == "__main__":
    main()
