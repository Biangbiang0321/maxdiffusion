# Causal Forcing Stage 1 Status

This branch implements a Causal Forcing Stage 1 style autoregressive Wan2.1 T2V path in MaxDiffusion.

## Implemented

- Strict local loading for `/home/bennett121358/ar_diffusion.pt`.
- Official-style AR outer loop over latent frame blocks.
- Conditional and unconditional self KV caches kept separately for CFG.
- Clean timestep-0 rerun after each generated block to update the self KV cache.
- Per-frame timestep tensors shaped `[batch, current_num_frames]`.
- RoPE temporal offset via `frame_start`.
- Official FlowUniPC timestep/sigma schedule alignment.
- Official Stage1 negative prompt in the causal config.
- Tokamax block-size handling for growing query/key lengths.
- Dense dot-product fallback for diagnostics.

## Verification

Focused tests:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src \
  /home/bennett121358/maxdiffusion_venv/bin/python -m pytest \
  src/maxdiffusion/tests/wan/wan_framewise_ar_inference_test.py \
  src/maxdiffusion/tests/wan/wan_causal_forcing_loader_test.py \
  src/maxdiffusion/tests/attention_test.py \
  -k "framewise_ar or causal_forcing or select_flash_block_sizes"
```

Result: `11 passed`.

Scheduler parity with official Causal Forcing `FlowUniPCMultistepScheduler`:

- Timesteps match exactly for 50 steps.
- Sigma max absolute difference: about `1.7e-7`.

Transformer parity with official PyTorch CausalWanModel on a tiny CPU input:

- First block forward cosine similarity: about `0.99988`.
- Strict checkpoint load: missing keys `0`, unexpected keys `0`.

TPU generation diagnostics:

- Original base Wan2.1 full-video path, 17 frames, 20 steps: clear toy robot video.
- Stage1 AR with `/home/bennett121358/ar_diffusion.pt`, `num_frame_per_block=1`, 17 frames, 50 steps: grid-like invalid video.
- Stage1 AR with `/home/bennett121358/ar_diffusion.pt`, `num_frame_per_block=3`, 9 frames, 50 steps: clear toy robot video.
- Stage1 AR with `/home/bennett121358/ar_diffusion.pt`, `num_frame_per_block=3`, 81 frames, 50 steps: clear usable toy robot video.

The clear 81-frame output was generated with:

```bash
/home/bennett121358/maxdiffusion_venv/bin/python src/maxdiffusion/generate_wan.py \
  src/maxdiffusion/configs/causal_forcing_framewise_ar.yml \
  prompt="A small toy robot walking across a desk" \
  framewise_ar_inference=True \
  framewise_ar_use_self_kv_cache=True \
  framewise_ar_num_frames_per_block=3 \
  framewise_ar_debug_stats=False \
  num_inference_steps=50 \
  num_frames=81 \
  attention=tokamax_flash
```

Output file:

```text
/home/bennett121358/maxdiffusion/wan_output_0_0.mp4
```

Contact sheet:

```text
/tmp/maxdiffusion_causal_frames/ar_chunk3_81f_50s_sheet.jpg
```

## Important Finding

The local checkpoint `/home/bennett121358/ar_diffusion.pt` behaves like a chunkwise Stage 1 checkpoint requiring:

```text
framewise_ar_num_frames_per_block=3
```

With `framewise_ar_num_frames_per_block=1`, the same checkpoint repeatedly produces grid-like invalid output after scheduler, mask, attention-kernel, text-negative-prompt, and VAE-path checks.

Therefore the exact framewise command in `goal.md` is not currently satisfiable with the only checkpoint present on disk. The implementation can run clear Stage 1 AR video with the available checkpoint when the block size is set to `3`.
