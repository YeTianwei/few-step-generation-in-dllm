# Project Log

## 2026-03-17

### Initial proxy milestone
- Added a new proxy sampler path for `text + action` masked infilling with a sampler-side coordination latent `z_c`.
- Kept the original `BD3LM` path unchanged and exposed the new sampler through `dllm.core.samplers`.
- Added a comparison script that runs the proxy task with coordination disabled/enabled on the same inputs.
- Added helper unit tests for layout partitioning, budget allocation, and coordination tensor shaping.

### Learnable `z_c` milestone
- Upgraded the proxy sampler so `z_c` can now be mediated by a trainable `CoordinationModule` instead of only heuristic cosine-based coupling.
- Implemented a zero-init residual design so the learned module starts near the old heuristic behavior and can be trained safely without immediately destabilizing decoding.
- Added save/load support for the coordination module so proxy-trained weights can be reused during inference.
- Added a lightweight proxy fitting script that freezes the backbone model and trains only the coordination module on masked text/action targets.
- Kept the inference API backward compatible: if no coordination checkpoint is provided, the sampler still works with its heuristic pathway.

### Files updated in this round
- `dllm/core/samplers/coord_proxy.py`: added `CoordinationModule`, lazy module initialization, differentiable coordination feature computation, and coordination checkpoint save/load hooks.
- `examples/a2d/bd3lm/coord_proxy_sample.py`: refactored into a safe `main()` entrypoint and kept the baseline vs coordinated comparison path.
- `examples/a2d/bd3lm/coord_proxy_fit.py`: added a minimal training path that freezes the backbone and optimizes only the coordination module.
- `scripts/tests/test_coord_proxy_sampler.py`: extended helper coverage to include coordination-module shape and zero-init behavior.

### Debug and first runnable checkpoint
- Fixed proxy marker parsing so `[TEXT_END]` and `[ACT_END]` can be found even when tokenization merges them with leading spaces or trailing newlines.
- Added a hidden-state fallback path for A2D Qwen checkpoints whose LM output exposes `logits` but not `hidden_states`.
- Fixed dtype mismatches between the `bfloat16` backbone and the newly added coordination module.
- Verified that `python examples/a2d/bd3lm/coord_proxy_sample.py` now runs end-to-end on `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`.
- First observed result: the proxy pipeline is runnable and the coordinated branch uses slightly fewer effective steps, but generation quality is still weak and needs proxy-task adaptation or coordination training before it is meaningful.

### Natural-language proxy adaptation experiment
- Replaced the original symbolic action proxy with a natural-language `Assistant response` plus `Action sequence` format to better match the pretrained text backbone.
- Changed prompt construction to operate directly at the token level: target sequences are built first, then only the text/action spans are replaced with mask tokens. This keeps prompt and target lengths exactly aligned for training and evaluation.
- Tightened evaluation to report token accuracy only on the generated text/action regions rather than over the whole sequence.
- Added a stronger learnable coordination pathway: the coordination module now predicts region-specific hidden deltas that are projected through the LM head into token-level logit adjustments.
- Ran a 30-epoch coordination-only proxy training run and evaluated the resulting checkpoint.
- Result: the setup trains stably in terms of loss reduction, but generation quality is still poor; the learned coordination module currently overconfidently collapses to low-quality punctuation-heavy completions and does not yet improve region token accuracy over the baseline.
