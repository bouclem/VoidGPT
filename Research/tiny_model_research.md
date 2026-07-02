# Tiny Intelligent Model Research

## Goal

Build a language model with ~120K parameters that is "a little intelligent" — not just a toy, but a model that can produce coherent, knowledgeable text.

## Key Findings

### Architecture (from research)

1. **MiniGPT (~96K params)** — Reference implementation
   - vocab=2000, d_model=32, n_heads=4, n_layers=2, d_ff=128, max_seq_len=128
   - Token embedding dominates at 67% of params
   - Weight tying between token embedding and output head

2. **Architectural trade-offs paper** (arxiv:2512.20877)
   - For ~430K params: 3 layers, embedding=128, 4 heads, d_ff=256 is optimal
   - Increasing depth without increasing optimization steps can degrade performance
   - Attention-based models offer strongest accuracy-efficiency trade-offs
   - RoPE may not improve performance at small scale (learned positional ok)

3. **Recursive transformer** (Chat-TRM / TinyRecursiveModels)
   - Weight reuse through recursion instead of stacking layers
   - Parameter efficient: ~0.5M-5M params with 2 blocks reused
   - Modern components: RMSNorm, SwiGLU, RoPE
   - Interesting for future iterations

### Data Strategy

1. **TinyHelen paper** (arxiv:2501.00522)
   - Tiny LMs need high quality, low noise, reduced complexity data
   - Curriculum learning helps: start simple, progress to complex
   - "Leaner" datasets: simple language, essential patterns only

2. **nano_wiki dataset** (HuggingFace: sixf0ur/nano_wiki)
   - Synthetic encyclopedia-style text, ~2.9M tokens
   - Simple English, controlled vocabulary
   - Designed for tiny LMs (<100M params)
   - Non-story, educational — perfect for VoidGPT 1 120K

3. **Simple Wikipedia** datasets
   - Shorter sentences, limited vocabulary
   - Broad general knowledge coverage
   - Good for Q&A and educational text

### Making Tiny Models Intelligent

1. **Knowledge distillation** — Train on outputs from larger models
2. **Curriculum learning** — Order data from simple to complex
3. **Data quality > quantity** — For tiny models, every token matters
4. **Simple language** — Short sentences, limited vocabulary work best
5. **Character-level tokenization** — Small vocab frees params for model capacity
6. **Weight tying** — Saves params by sharing embedding and output weights

## VoidGPT 1 120K Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tokenizer | Character-level | Small vocab (~100) → more params for model brain |
| d_model | 64 | Balance between capacity and param budget |
| n_heads | 4 | head_dim=16, good for small models |
| n_layers | 2 | Optimal for ~120K params (more layers = undertrained) |
| d_ff | 256 | 4x expansion, standard ratio |
| Positional | Learned | Simpler, works well at small scale |
| Normalization | LayerNorm (pre-norm) | Stable training |
| Activation | GELU | Standard, smooth gradients |
| Weight tying | Yes | Saves ~8K params |
| Data | Encyclopedia-style | Non-story, educational, simple English |

## Parameter Budget (~120K)

| Component | Parameters | % of Total |
|-----------|-----------|------------|
| Token Embedding (tied) | 6,400 | 5.3% |
| Positional Embedding | 8,192 | 6.8% |
| Transformer Block ×2 | ~99,968 | 83.3% |
| Final LayerNorm | 128 | 0.1% |
| Output Bias | ~100 | 0.8% |
| **Total** | **~116K** | **100%** |

Note: Character-level vocab (100 chars) makes embedding small, so most params go to the transformer body — this is ideal for intelligence.

## Future Research Directions

- [x] RMSNorm (simpler than LayerNorm, fewer params) — **implemented as option**
- [x] Recursive weight reuse (parameter efficiency) — **implemented as recursion_steps**
- [x] SwiGLU activation (better than GELU, used in LLaMA) — **implemented as option**
- [x] RoPE positional encoding (better generalization to longer sequences) — **implemented as option**
- [ ] Knowledge distillation from larger models
- [ ] Curriculum learning implementation
- [ ] BPE tokenizer for larger vocab experiments
- [ ] Hybrid Transformer+Mamba layers (alternating attention and SSM)
- [ ] LoRA adapters per recursion step (layer specialization, Google DeepMind approach)

## Architecture Alternatives Research (Iteration 1 Update)

### Why Not Just Transformer?

For ~120K params, a pure decoder-only transformer works but has limitations:
- Fixed depth — can't "think more" without adding params
- No mechanism for iterative refinement

### Recursive Weight Reuse (IMPLEMENTED)

**Sources**: Google DeepMind "Relaxed Recursive Transformers", Samsung "Tiny Recursive Models"

**Key idea**: Apply the same transformer blocks N times instead of stacking N×more blocks.
- 2 blocks × 3 recursion steps = effective depth 6, but only 2 blocks worth of params
- Google DeepMind showed recursive models converted from 2× larger models can beat same-size pretrained models
- Samsung TRM achieved 45% on ARC-AGI-1 with only 7M params using recursive reasoning

**Our implementation**: `recursion_steps` config parameter. Default=1 (standard), set to 2-3 for recursive mode.

### Hybrid Transformer+Mamba

**Sources**: Jamba (AI21), Zamba2 (Zyphra), TransMamba (AAAI 2026)

**Key idea**: Alternate between attention layers (good at retrieval) and Mamba SSM layers (linear time, good for long sequences).
- Zamba2-2.7B: Mamba backbone + shared attention blocks, state-of-the-art <3B
- At 120K params, Mamba's selective scan overhead may not pay off
- Better suited for larger models or very long sequences

**Decision**: Not implemented yet — complexity too high for 120K param budget. Track for future iterations.

### Mamba/SSM Only

**Source**: Mamba (arxiv:2312.00752)

**Key idea**: Selective state space models replace attention entirely. Linear time complexity.
- Good for long sequences but at 128 token context, advantage is minimal
- Requires CUDA-optimized selective scan for good performance
- Pure PyTorch fallback is slow

**Decision**: Not practical at 120K params with 128 context length.

### Real Datasets (IMPLEMENTED)

Added support for real HuggingFace datasets:
- **nano_wiki**: 9,107 synthetic encyclopedia articles, ~2.9M tokens, simple English
- **WikiText-2**: Real Wikipedia articles, verified quality
- **Simple Wikipedia Q&A**: 433K Q&A pairs from Simple English Wikipedia
- **Mixed mode**: Synthetic knowledge + nano_wiki combined
