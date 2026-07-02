# Changelog

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
