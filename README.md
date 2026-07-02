# VoidGPT

A series of AI models built from scratch, part of the VoidGPT project family.

## Structure

```
VoidGPT/
├── Research/       # Research notes and experiments for tiny intelligent models
├── VoidGPT1/       # VoidGPT Generation 1 — tiny transformer models
│   └── 120K/       # 120K parameter model (first optimized version)
└── docs/           # Project documentation
```

## VoidGPT 1 120K

The first version of VoidGPT 1, optimized for ~120K parameters.

- **Architecture**: Decoder-only transformer with recursive weight reuse
- **Tokenizer**: Character-level
- **Parameters**: ~120K (same param count with configurable recursion depth)
- **Normalization**: LayerNorm (default) or RMSNorm (optional)
- **Training data**: Non-story educational text — synthetic knowledge entries + real datasets (nano_wiki, WikiText-2, Simple Wikipedia Q&A)
- **Goal**: A small but "little intelligent" model — optimized, not just small

### Key Features

- **Recursive weight reuse**: Apply transformer blocks N times with shared weights for more "thinking depth" without adding parameters (`--recursion_steps`)
- **RMSNorm option**: Simpler normalization with fewer parameters (`--rmsnorm`)
- **Real datasets**: Load from HuggingFace — nano_wiki, WikiText-2, Simple Wikipedia Q&A, or mixed mode
- **Weight tying**: Token embedding and output head share weights

See `VoidGPT1/120K/` for implementation details.
