# Changelog

## [0.5.0] — 2026-07-02

### Results
- Full 5000-iter training with best variant (RoPE+SwiGLU+RMSNorm+Recursive3)
- Final val PPL: 1.17 (near-perfect on synthetic data)
- Training time: 3:49 on CUDA, 178K tok/s
- Model generates coherent factual text: "Water is a giant magnet. The center of an atom is called the nucleus."
- 104,040 params (104.0K) — well within 120K budget

## [0.4.0] — 2026-07-02

### Added
- `benchmark.py`: Architecture variant comparison script (7 variants)
- Loss curve visualization: 3 matplotlib charts (val loss, val PPL, final comparison)
- `matplotlib>=3.7.0` to requirements.txt
- Benchmark results: all+recursive(3) is best variant (PPL 8.83 vs standard 10.84)

### Key Findings
- RoPE is the single biggest improvement: PPL 10.84→9.42 (-13%)
- Recursive depth (3 steps) adds another -6% PPL
- All features combined: -18.5% PPL with 7.6% fewer params (104K vs 112.6K)

## [0.3.0] — 2026-07-02

### Added
- SwiGLU FFN: `(SiLU(xW1) * xW2) W3` — modern activation used in LLaMA, PaLM, Mistral
- RoPE (Rotary Position Embedding): removes learned positional embeddings, saves ~8K params
- `--swiglu` and `--rope` CLI flags in train.py
- `rope_theta` config parameter (default 10000.0)
- Pytest test suite: 27 tests covering all model features (test_model.py)
- SwiGLU, RoPE, and all-features tests in verify.py

### Fixed
- Missing `unk_id` property on `CharTokenizer`

### Changed
- `CausalSelfAttention` now optionally applies RoPE to Q and K
- `VoidGPT120K` conditionally creates positional embeddings (None when using RoPE)
- `TransformerBlock` selects between `FeedForward` and `SwiGLUFFN` based on config

## [0.2.0] — 2026-07-02

### Added
- `.gitignore` for Python, checkpoints, data, and IDE files
- Real-time training display: progress %, running loss, PPL, tokens/sec, ETA
- Perplexity (PPL) metric in eval and final summary
- Training summary: total time, total tokens, avg speed
- Checkpoints now save `val_ppl` alongside `val_loss`

### Changed
- Training output uses carriage return for live-updating progress line
- Eval output formatted with box-drawing characters for readability

## [0.1.0] — 2026-07-02

### Added
- VoidGPT 1 120K: Initial model implementation
  - Character-level tokenizer
  - Decoder-only transformer (~120K params)
  - Training pipeline with AdamW, cosine LR schedule, gradient clipping
  - Text generation with temperature/top-k sampling
  - Non-story educational dataset preparation
- Research folder with notes on tiny intelligent model design
