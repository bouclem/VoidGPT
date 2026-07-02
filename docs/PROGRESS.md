# VoidGPT Progress Log

## Iteration 1 — Initial Build (2026-07-02)

### Phase 1: PLAN
- **Feature**: Build the complete VoidGPT 1 120K model from scratch
- **Why**: First version of the VoidGPT 1 series. Must be optimized and "a little intelligent."
- **Architecture**: Decoder-only transformer, ~120K params
  - Character-level tokenizer (small vocab → more params for model brain)
  - d_model=64, n_heads=4, n_layers=2, d_ff=256
  - Pre-norm LayerNorm, GELU FFN, learned positional embeddings
  - Weight tying (token emb = output head)
- **Data**: Non-story educational text (encyclopedia-style, simple English)
- **Files**: config.py, tokenizer.py, model.py, dataset.py, train.py, generate.py, prepare_data.py, requirements.txt

### Phase 2: RESEARCH
- Studied MiniGPT (~96K params) — similar scale, good reference architecture
- Studied TinyHelen paper — curriculum learning for tiny LMs, low-noise/low-complexity data
- Found nano_wiki dataset — synthetic encyclopedia, ~2.9M tokens, simple English, designed for tiny LMs
- Architectural trade-offs paper: 3 layers optimal for ~430K, 2 layers better for ~120K
- Recursive transformer approach (weight reuse) — interesting for future iterations
- Key insight: For tiny models, data quality and simplicity matter more than architecture complexity

### Phase 3: CODE — COMPLETED
- All files created: config.py, tokenizer.py, model.py, dataset.py, train.py, generate.py, verify.py, requirements.txt
- data/prepare_data.py with synthetic + real dataset support
- .gitignore added
- User feedback incorporated: recursive weight reuse, RMSNorm, real datasets, PPL, real-time training

### Phase 4: VERIFY — PASSED
- verify.py: All tests passed (tokenizer, standard model, recursive model, RMSNorm, forward pass, generation)
- Param count: 112,640 (112.6K) — within 120K budget
- Training smoke test (200 iters): loss 4.25→2.83, PPL 70→17, works on CUDA
- Recursive training test (100 iters, 3 recursion steps): loss 4.24→3.53, same param count
- Generation test: works (gibberish at 200 iters, expected)

### Phase 5: FIX — No issues found

### Phase 6: COMPLETE & IMPROVE
- **Accomplished**: Full VoidGPT 1 120K model from scratch with recursive weight reuse, RMSNorm option, real dataset support, real-time training with PPL
- **Learned**: At 120K params, character-level tokenizer is ideal (small vocab → more params for model body). Recursive weight reuse gives free depth. Real-time training display greatly improves UX.
- **Next iteration**: SwiGLU activation (better than GELU), RoPE positional encoding, proper test suite with pytest, longer training run to evaluate intelligence

---

## Iteration 2 — Architecture Improvements (2026-07-02)

### Phase 1: PLAN
- **Feature**: Add SwiGLU activation and RoPE positional encoding + pytest test suite
- **Why**: SwiGLU (used in LLaMA, PaLM) outperforms GELU. RoPE enables better generalization to longer sequences and removes positional embedding params.
- **Files to modify**: config.py, model.py, verify.py
- **Files to create**: test_model.py, test_tokenizer.py

### Phase 2: RESEARCH
- SwiGLU: `FFN(x) = (SiLU(xW1) * xW2) W3` — 3 linear layers, inner dim (2/3)*d_ff to match param count
- RoPE: Rotate Q,K pairs by `m*theta_i` where `theta_i = theta^(-2i/d)`. Position 0 = identity. No learned params.
- Both used in modern LLMs (LLaMA, Mistral, PaLM, Gemma)
- Pytest: fixtures, parametrized tests, class-based test organization

### Phase 3: CODE — COMPLETED
- config.py: Added `use_swiglu`, `use_rope`, `rope_theta` config options
- model.py: Added `RoPE` class, `SwiGLUFFN` class, updated `CausalSelfAttention` with optional RoPE, updated `VoidGPT120K` to conditionally use RoPE
- train.py: Added `--swiglu` and `--rope` CLI flags
- verify.py: Added SwiGLU, RoPE, and all-features tests
- test_model.py: 27 pytest tests covering all model features

### Phase 4: VERIFY — PASSED
- verify.py: All 8 test groups passed
- pytest: 27/27 tests passed in 2.23s
- Param counts:
  - Standard: 114,688 (114.7K)
  - SwiGLU: 114,600 (114.6K)
  - RoPE: 106,496 (106.5K) — saves 8,192 params from pos emb
  - All features: 106,088 (106.1K)
- Training smoke test (200 iters, all features): val PPL 16.78 vs standard 16.91 — improvement with fewer params

### Phase 5: FIX
- Fixed: `CharTokenizer` missing `unk_id` property — added property returning `SPECIAL_TOKENS["<UNK>"]`
- After fix: 27/27 tests pass

### Phase 6: COMPLETE & IMPROVE
- **Accomplished**: SwiGLU FFN, RoPE positional encoding, comprehensive pytest suite (27 tests)
- **Learned**: RoPE saves ~8K params by removing learned positional embeddings. SwiGLU + RoPE + RMSNorm combined gives better PPL with fewer params. Pytest class-based organization works well for feature-grouped tests.
- **Next iteration**: Longer training run (5000 iters) to evaluate real intelligence, compare architecture variants, add training loss curve visualization

---

## Iteration 3 — Benchmark & Visualization (2026-07-02)

### Phase 1: PLAN
- **Feature**: Architecture variant benchmark script + loss curve visualization
- **Why**: Need to empirically determine which architecture combination performs best
- **Files to create**: benchmark.py
- **Files to modify**: requirements.txt (add matplotlib), .gitignore

### Phase 2: RESEARCH
- Matplotlib: standard approach is collect loss history in lists, plot with plt.plot, save with plt.savefig
- Benchmark pattern: train each variant with same data/hyperparams, compare final val PPL

### Phase 3: CODE — COMPLETED
- benchmark.py: 7 architecture variants, trains each, plots 3 charts (val loss, val PPL, final comparison bar chart)
- requirements.txt: Added matplotlib>=3.7.0
- .gitignore: Updated checkpoint patterns, keep benchmark plots

### Phase 4: VERIFY — PASSED
- Benchmark ran successfully on CUDA, 7 variants × 500 iters
- Results:
  - standard:        112,640 params, PPL 10.84
  - rmsnorm:         112,320 params, PPL 10.91
  - swiglu:          112,552 params, PPL 10.87
  - rope:            104,448 params, PPL  9.42
  - rope+swiglu+rmsnorm: 104,040 params, PPL  9.38
  - recursive(3):    112,640 params, PPL 10.81
  - all+recursive(3): 104,040 params, PPL  8.83 ← BEST
- Plot saved to benchmark_plot.png (3 charts)

### Phase 5: FIX — No issues found

### Phase 6: COMPLETE & IMPROVE
- **Accomplished**: Benchmark script with 7 variants, loss curve visualization, empirical evidence that RoPE+SwiGLU+RMSNorm+Recursive is best
- **Key finding**: RoPE is the single biggest improvement (PPL 10.84→9.42, -13%). Recursive depth adds another -6%. All features combined: -18.5% PPL with 7.6% fewer params.
- **Learned**: At 120K scale, positional encoding choice matters more than activation function. RoPE's parameter savings + better generalization is a double win.
- **Next iteration**: Full 5000-iter training run with best variant, generate sample text to evaluate intelligence, add sample generation to benchmark

### User Feedback (mid-iteration)
- User asked: "use real datasets too, and why transformer only? for intelligent transformer not adapted?"
- Researched architecture alternatives: recursive transformers, hybrid Transformer+Mamba, Mamba/SSM
- Researched real datasets: nano_wiki, WikiText-2, Simple Wikipedia Q&A

### Architecture Updates
- **Recursive weight reuse**: Added `recursion_steps` config — same blocks applied N times for more depth without more params
  - Based on Google DeepMind "Relaxed Recursive Transformers" and Samsung "Tiny Recursive Models"
- **RMSNorm**: Added as optional normalization (fewer params than LayerNorm)
- Decided against Mamba/SSM at 120K scale — complexity not justified for 128-token context

### Dataset Updates
- Added real dataset loading via HuggingFace `datasets` library:
  - `nano_wiki`: 9,107 synthetic encyclopedia articles, simple English, ~2.9M tokens
  - `wikitext2`: Real Wikipedia articles, verified quality
  - `simple_wiki_qa`: 433K Q&A pairs from Simple English Wikipedia
  - `mixed`: Synthetic knowledge + nano_wiki combined
- `prepare_data.py` now supports `--source` flag for dataset selection
