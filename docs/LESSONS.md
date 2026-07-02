# Lessons Learned — VoidGPT 1 120K Development

Cumulative file of mistakes, patterns, and insights learned across all iterations.

## Iteration 1-2
- **Character-level tokenizer is ideal for tiny models**: Small vocab (~100 chars) means embedding layer is small, freeing params for the transformer body
- **Recursive weight reuse is free depth**: Same blocks applied N times = N× effective depth with 0 extra params
- **PowerShell mangles inline Python quotes**: Never use complex one-liner Python commands in PowerShell. Write a script file instead.
- **write_to_file fails on existing files**: Use multi_edit for targeted changes to existing files

## Iteration 3
- **RoPE is the biggest single improvement** at 120K scale: PPL -13% from positional encoding alone
- **Width > depth at tiny scale**: d96_l1_rec3 (120K, 3 effective layers) beats d64_l2_rec3 (105K, 6 effective layers)
- **Benchmark before long training**: Short 500-iter benchmarks save time vs full 5000-iter runs for architecture comparison

## Iteration 4
- **PPL 1.17 on synthetic data = memorization**: Need real datasets to evaluate generalization
- **Train/val gap indicates generalization**: PPL 3.56 train vs 3.94 val on nano_wiki = healthy generalization

## Iteration 5
- **Automatic loss curve plots are essential**: matplotlib PNG saved after training gives quick visual feedback
- **Real data gives realistic PPL**: nano_wiki PPL 3.94 vs synthetic PPL 1.17

## Iteration 6
- **eval_iters=200 causes training lag**: 400 forward passes (200 train + 200 val) every 500 iters freezes terminal for seconds. Reduced to 25 — 8x faster with stable estimates.
- **Async checkpoint saving**: GPU→CPU copy + torch.save must happen in background thread to avoid blocking training loop
- **'evaluating...' indicator**: Show text during eval so terminal doesn't appear frozen

## Iteration 7
- **Repeated synthetic data causes overfitting**: Synthetic entries repeated 50× dominate training signal. Mixed dataset PPL 5.46 vs pure nano_wiki PPL 3.84.
- **Data quality > data quantity for tiny models**: Diverse real text generalizes better than repeated structured facts
- **Train/val gap reveals overfitting**: PPL 2.18 train vs 5.46 val = severe overfitting. Healthy gap is ~0.3-0.5 PPL.
- **More iters ≠ better when overfitting**: Best val PPL at iter 6000, then plateaued. Early stopping or lower LR would help.

## Iteration 8
- **Higher dropout + weight decay prevents overfitting**: dropout=0.2, weight_decay=0.2 kept PPL improving through 5000 iters with no plateau
- **Vocab size matters for param budget**: 87 chars = 8.8K embedding params, 180 chars = 17.3K. That's 8.5K fewer params for transformer body at 120K scale.
- **Diverse real data > repeated synthetic**: nano_wiki+wikitext2 (PPL 7.12, no overfit) vs synthetic+nano_wiki (PPL 5.46, overfit at 6000)
- **Larger vocab = harder problem**: PPL 7.12 on 180 chars vs PPL 3.84 on 87 chars. Not directly comparable — more chars = more uncertainty per position.
