# Goal: Reproduce Causal Forcing Stage 1 in MaxDiffusion

本文档记录 `causal_attention` 分支的真实目标：在当前 MaxDiffusion 代码库中复现 Causal Forcing Stage 1 的结构和功能，并使用对应的 Stage 1 模型参数生成一个清晰、可用的视频。

重点不是“完成一个 mask”，也不是单独实现某个 attention kernel。mask、AR loop、KV cache、checkpoint loading、scheduler 参数、CFG 逻辑都只是实现手段。最终验收必须落在：代码结构接近官方 Stage 1，参数正确加载，目标 T2V 命令能跑通，并输出质量合理的视频。

## Objective

在 MaxDiffusion 的 Wan2.1 T2V 路径中复现官方 Causal Forcing Stage 1 `Autoregressive Diffusion` 的核心结构和推理功能：

- 使用 Wan2.1 T2V 1.3B 作为基础模型。
- 使用 Causal Forcing Stage 1 framewise AR diffusion checkpoint：

```text
/home/bennett121358/ar_diffusion.pt
```

- 输入 text prompt，输出 81 帧 T2V 视频。
- 模型结构和推理方式应对应官方 Stage 1 framewise AR diffusion，而不是原始 Wan2.1 bidirectional full-video denoise。
- 代码结构应保留官方 `num_frame_per_block` 设计：`1` 表示 framewise AR，`>1` 表示 chunkwise / multi-frame AR；不能把实现硬编码成只支持单帧。
- 目标视频应该是清晰、可辨认、和 prompt 对齐的结果；不是纯噪声、灰图、严重糊图、崩坏运动或明显由结构不匹配导致的错误输出。

目标运行命令：

```bash
python src/maxdiffusion/generate_wan.py \
  src/maxdiffusion/configs/causal_forcing_framewise_ar.yml \
  prompt="A small toy robot walking across a desk" \
  framewise_ar_inference=True \
  framewise_ar_use_self_kv_cache=True \
  framewise_ar_num_frames_per_block=1 \
  num_inference_steps=50
```

## Official Reference

官方 Causal Forcing Stage 1 是 `Autoregressive Diffusion Training`。官方用于测试 Stage 1 训练结果的命令是：

```bash
python inference.py \
  --config_path configs/ar_diffusion_tf_{framewise OR chunkwise}.yaml \
  --output_folder output/{framewise OR chunkwise}_ar_diffusion \
  --checkpoint_path checkpoints/{framewise OR chunkwise}/ar_diffusion.pt \
  --data_path prompts/demos.txt
```

该命令默认是 T2V，因为没有传 `--i2v`，`data_path` 是 prompt 文本文件。

本分支对齐的是：

- `configs/ar_diffusion_tf_framewise.yaml`
- `checkpoints/framewise/ar_diffusion.pt`
- framewise Stage 1 AR diffusion inference
- T2V usage

本分支不以这些为目标：

- `causal_forcing.pt`
- `causal_forcing_dmd_framewise.yaml`
- Stage 2 Causal ODE / Causal CD
- Stage 3 DMD few-step model
- Causal Forcing++ 1-step / 2-step model
- I2V demo

## Official Code Constraints

实现必须以官方 `thu-ml/Causal-Forcing` 代码结构为依据，不能只按直觉写一个“看起来 causal”的版本。

官方 Stage 1 同时支持两种 AR block 设置：

- framewise AR: `num_frame_per_block: 1`
- chunkwise / multi-frame AR: `num_frame_per_block: 3`

官方配置里：

```yaml
# configs/ar_diffusion_tf_framewise.yaml
num_frame_per_block: 1

# configs/ar_diffusion_tf_chunkwise.yaml
num_frame_per_block: 3
```

官方 `CausalDiffusionInferencePipeline` 会读取 `num_frame_per_block`，并用它构造 AR 外层循环：

```python
all_num_frames = [self.num_frame_per_block] * num_blocks
for current_num_frames in all_num_frames:
  noisy_input = noise[:, ... : ... + current_num_frames]
  latents = noisy_input
  for t in sample_scheduler.timesteps:
    flow_pred_cond = generator(latents, timestep=[B, current_num_frames], kv_cache=kv_cache_pos)
    flow_pred_uncond = generator(latents, timestep=[B, current_num_frames], kv_cache=kv_cache_neg)
    flow_pred = flow_pred_uncond + guidance_scale * (flow_pred_cond - flow_pred_uncond)
    latents = scheduler.step(flow_pred, t, latents)
  output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents
  generator(latents, timestep=0, kv_cache=kv_cache_pos)
  generator(latents, timestep=0, kv_cache=kv_cache_neg)
```

也就是说：

- `num_frame_per_block=1` 是单 latent frame 自回归。
- `num_frame_per_block>1` 是多 latent frame / chunk 自回归。
- 每个 block 结束后，官方会用 clean latent 和 timestep 0 再跑一次 generator 来更新 KV cache。
- conditional 和 unconditional 有独立 KV cache：`kv_cache_pos` 和 `kv_cache_neg`。

官方模型里有 blockwise causal mask：

```python
ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block
allow = kv_idx < ends[q_idx] or q_idx == kv_idx
```

这个 mask 的语义是 blockwise causal：

- block 之前的 tokens 可见。
- 当前 block 内 tokens 互相可见。
- 未来 block 不可见。
- 当 `num_frame_per_block=1` 时，它退化成 framewise causal。
- 当 `num_frame_per_block=3` 时，它是 3-frame chunkwise causal，不是严格逐帧 causal。

官方推理的 KV-cache path 不是靠每步构造 full-video dense mask 来工作，而是：

- 当前 block 作为 query 输入。
- 当前 block 的 K/V 写进 cache。
- attention 读取历史 cache 加当前 block cache。
- 用 `current_start` 和 `cache_start` 控制 RoPE/frame position 和 cache 写入位置。

因此 MaxDiffusion 复现时应该先对齐这个结构，再考虑 JAX/TPU 下的实现细节。

## Definition Of Done

任务只有在以下条件都满足时才算完成：

- 目标命令能在 TPU 上完整跑完。
- 输出 mp4 成功写入 output directory。
- 输出视频肉眼可辨认：能看出 toy robot、desk、walking/motion 语义。
- 视频不是明显的结构错误结果，例如：
  - 全噪声
  - 全灰或全黑
  - 严重糊成不可辨认物体
  - 帧间完全不连贯
  - 明显和 prompt 无关
- `/home/bennett121358/ar_diffusion.pt` 被正确加载到 Wan transformer，而不是静默退回 base Wan 权重。
- 推理结构是 framewise AR：21 个 latent frames 按顺序生成，不是 21 个 latent frames 一次性 bidirectional denoise。
- 当前 frame 只能依赖 text context 和历史 clean video context，不能依赖未来 video frames。
- CFG、scheduler、flow shift、negative prompt、latent/video shape 与官方 Stage 1 配置保持合理一致。

## Current Status Update

当前实现已经能用 `/home/bennett121358/ar_diffusion.pt` 跑通 Causal Forcing Stage 1 风格的 chunkwise AR 生成，并输出清晰可用的 81 帧 T2V 视频。

实测可用命令是：

```bash
/home/bennett121358/maxdiffusion_venv/bin/python src/maxdiffusion/generate_wan.py \
  src/maxdiffusion/configs/causal_forcing_framewise_ar.yml \
  prompt="A small toy robot walking across a desk" \
  framewise_ar_inference=True \
  framewise_ar_use_self_kv_cache=True \
  framewise_ar_num_frames_per_block=3 \
  num_inference_steps=50 \
  num_frames=81 \
  attention=tokamax_flash
```

输出文件：

```text
/home/bennett121358/maxdiffusion/wan_output_0_0.mp4
```

抽帧检查图：

```text
/tmp/maxdiffusion_causal_frames/ar_chunk3_81f_50s_sheet.jpg
```

关键发现：

- `framewise_ar_num_frames_per_block=3` 生成清晰 toy robot 视频。
- `framewise_ar_num_frames_per_block=1` 在 17 帧和此前 81 帧诊断中都生成规则网格，不是清晰视频。
- base Wan2.1 full-video 路径能正常生成清晰 toy robot，因此 VAE、text encoder、视频保存路径不是根因。
- dot-product attention 和 Tokamax attention 下 block=1 都是网格，因此不是 Tokamax 单独导致。
- scheduler 已和官方 `FlowUniPCMultistepScheduler` 对齐，50-step timesteps 完全一致，sigma 误差约 `1.7e-7`。

因此，当前唯一 checkpoint `/home/bennett121358/ar_diffusion.pt` 实测更像官方 chunkwise Stage 1 checkpoint，而不是 framewise checkpoint。若 Definition Of Done 必须坚持 `framewise_ar_num_frames_per_block=1`，还需要一个真正匹配 framewise 配置的 checkpoint；否则当前代码和 checkpoint 的可用运行方式应使用 `framewise_ar_num_frames_per_block=3`。

## Success Criteria

### Functional

- `framewise_ar_inference=True` 时进入 Stage1-like AR inference path。
- 81 pixel frames 经 Wan VAE 对应 21 latent frames。
- `framewise_ar_num_frames_per_block=1` 时，AR 外层循环逐个 latent frame 生成。
- 每个 latent frame/block 有自己的 denoise trajectory。
- 历史 clean frames 不会被后续 scheduler step 改写。
- 当前 frame denoise 后写回 `generated_latents`。
- 完成当前 frame 后，更新历史 context，供后续 frame 使用。

### Model / Checkpoint

- 确认 `/home/bennett121358/ar_diffusion.pt` 的参数结构和当前 MaxDiffusion Wan transformer 参数结构匹配。
- 如果 checkpoint conversion 或 key mapping 不正确，必须修复，不能只靠 base model 跑通。
- 生成质量差时，应优先排查 checkpoint 是否正确加载、RoPE/timestep/attention/scheduler 是否对齐，而不是只调 prompt。

### Attention / Causality

- 官方通用 blockwise causal self-attention 语义：

```python
query_block = query_frame_id // num_frame_per_block
key_block = key_frame_id // num_frame_per_block
allow = key_block < query_block or key_block == query_block
```

- 对 `num_frame_per_block=1`，它退化为 framewise causal：

```python
allow = key_frame_id <= query_frame_id
```

- 对 `num_frame_per_block>1`，当前 block / chunk 内是 bidirectional visibility，未来 block 不可见。
- mask 只是实现 causal visibility 的手段，不是最终目标。
- 如果使用 self KV cache，当前 block forward 只输入当前 block tokens，历史 tokens 来自 KV cache；block 内 visibility 由当前 block 的 query/K/V 共同输入保证。
- 如果不用 self KV cache，则需要 full-video fallback + per-token timestep + blockwise causal mask。

### CFG

- CFG 合成必须保持：

```python
noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
```

- CFG 下 self KV cache 应分开维护：

```python
self_kv_cache_cond
self_kv_cache_uncond
```

- 不应维护一个巨大的 `2 * batch` self KV cache buffer。

### TPU / Kernel

- Tokamax/Splash attention 不应触发：

```text
bkv_compute must be a multiple of 128
block_kv=11008 compile-time vmem OOM
append_self_kv_cache single-buffer 5GB HBM OOM
```

- KV cache 变长时，Tokamax block size 应保持合理分块，例如当前配置下：

```text
block_kv=512
block_kv_compute=512
```

## Implementation Areas

### 1. Config Alignment

目标文件：

```text
src/maxdiffusion/configs/causal_forcing_framewise_ar.yml
```

需要确认：

```yaml
model_name: wan2.1
model_type: T2V
pretrained_model_name_or_path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
wan_transformer_pretrained_model_name_or_path: /home/bennett121358/ar_diffusion.pt
framewise_causal_attention: True
framewise_ar_inference: True
framewise_ar_num_frames_per_block: 1
framewise_ar_use_self_kv_cache: True
flow_shift: 5.0
guidance_scale: 3.0
num_inference_steps: 50
num_frames: 81
height: 480
width: 832
fps: 16
```

### 2. Checkpoint Loading

目标文件可能包括：

```text
src/maxdiffusion/checkpointing/*
src/maxdiffusion/generate_wan.py
src/maxdiffusion/models/wan/transformers/transformer_wan.py
```

需要确认：

- `ar_diffusion.pt` 确实被读取。
- checkpoint key mapping 正确。
- transformer 参数被替换为 Stage 1 AR diffusion 参数。
- dtype 和 shape 对齐。
- 加载失败时应报错或明确记录，不能静默使用 base weights。

### 3. Stage1-like AR Inference

目标文件：

```text
src/maxdiffusion/pipelines/wan/wan_pipeline_2_1.py
src/maxdiffusion/pipelines/wan/wan_framewise_ar_utils.py
```

核心结构：

```python
generated_latents = initial_noise
history_context = empty

for block_start in range(0, latent_num_frames, num_frames_per_block):
  current_latents = generated_latents[:, :, block_start : block_start + num_frames_per_block]
  frame_scheduler_state = scheduler.set_timesteps(...)

  for t in timesteps:
    noise_pred = transformer(
        current_latents,
        text_context,
        history_context,
        timestep=t,
    )
    current_latents = scheduler.step(noise_pred, t, current_latents)

  generated_latents = replace_latent_frame_block(
      generated_latents, current_latents, block_start, num_frames_per_block)
  history_context = update_history_context(current_latents)
```

### 4. Blockwise / Framewise Causal Self-Attention

目标文件：

```text
src/maxdiffusion/kernels/splash_attention/splash_attention_mask.py
src/maxdiffusion/models/attention_flax.py
src/maxdiffusion/models/wan/transformers/transformer_wan.py
```

需要确认：

- full-video fallback 使用官方 blockwise causal mask。
- `num_frame_per_block=1` 时 mask 是 framewise causal。
- `num_frame_per_block>1` 时 mask 是 chunkwise causal，当前 chunk 内可互相 attention。
- self KV cache path 不看未来 frame。
- attention shape、RoPE slice、tokens-per-frame 计算正确。

### 5. Self KV Cache

目标文件：

```text
src/maxdiffusion/models/attention_flax.py
src/maxdiffusion/models/wan/transformers/transformer_wan.py
src/maxdiffusion/pipelines/wan/wan_pipeline.py
src/maxdiffusion/pipelines/wan/wan_pipeline_2_1.py
src/maxdiffusion/pipelines/wan/wan_framewise_ar_utils.py
```

需要确认：

- 每层 self-attention 可以返回当前 frame 的 post-RoPE K/V。
- cache shape 约为：

```text
[layers, batch, heads, cached_tokens, head_dim]
```

- 追加轴是 sequence axis。
- 后续 frame 使用历史 clean K/V + 当前 K/V。
- CFG cond/uncond cache 分开维护。

### 6. Scheduler / Timestep / RoPE

需要重点核对：

- `flow_shift: 5.0`
- `guidance_scale: 3.0`
- `num_inference_steps: 50`
- `timestep` shape 是否和 transformer 期望一致。
- per-frame timestep 是否符合 AR diffusion 推理语义。
- `current_rotary_emb` 是否只对应当前 latent frame tokens。
- full-video fallback 下历史 frame timestep 是否为 0。

## Validation

### Focused Tests

```bash
python -m pytest src/maxdiffusion/tests/wan/wan_framewise_ar_inference_test.py
python -m pytest src/maxdiffusion/tests/wan/wan_framewise_mask_test.py
python -m pytest src/maxdiffusion/tests/attention_test.py -k "select_flash_block_sizes or tokamax_cached_self_attention"
```

### Syntax / Import

```bash
PYTHONPATH=src python -c "import maxdiffusion"
python -m py_compile src/maxdiffusion/pipelines/wan/wan_pipeline_2_1.py
python -m py_compile src/maxdiffusion/models/attention_flax.py
```

### End-to-End TPU Run

```bash
python src/maxdiffusion/generate_wan.py \
  src/maxdiffusion/configs/causal_forcing_framewise_ar.yml \
  prompt="A small toy robot walking across a desk" \
  framewise_ar_inference=True \
  framewise_ar_use_self_kv_cache=True \
  framewise_ar_num_frames_per_block=1 \
  num_inference_steps=50
```

Expected:

- Command finishes.
- Output mp4 exists.
- Output video is visually inspected.
- Video is reasonably clear and prompt-aligned.
- Any generated failure is debugged as a Stage 1 reproduction mismatch, not treated as success merely because the command exited.

### Quality Check

Use the generated video to answer:

- Can I clearly identify a small toy robot?
- Is it on or crossing a desk?
- Is there visible motion across frames?
- Are the frames temporally coherent?
- Is the video much better than the previous broken full-video causal-mask-only output?

If the answer is no, the task is not complete even if tests pass.

## Known Risks

- Official Stage 1 AR diffusion may still be lower quality than final Stage 3 DMD models, but it should not be broken or unreadable.
- Growth-style JAX KV cache may trigger multiple XLA compiles because cache length changes by frame.
- Later latent frames may still OOM if total history KV cache becomes too large.
- Checkpoint mapping may be incomplete or silently wrong.
- Scheduler/timestep/RoPE mismatch can produce blurry or incorrect video even if shapes pass.

If OOM continues, next options:

- fixed-shape KV cache
- segmented KV cache
- lower resolution for debugging
- temporarily disable `framewise_ar_use_self_kv_cache`

If quality is poor, next checks:

- checkpoint key mapping and loaded parameter count
- whether base Wan weights are accidentally used
- timestep schedule and flow shift
- RoPE frame slicing
- negative prompt and CFG order
- causal history semantics
- VAE decode path and latent layout

## Progress Checklist

- [ ] Confirm official Stage 1 reference behavior and config fields.
- [ ] Verify `/home/bennett121358/ar_diffusion.pt` key structure.
- [ ] Verify Stage 1 checkpoint is actually loaded into Wan transformer.
- [ ] Align config with official `ar_diffusion_tf_framewise.yaml`.
- [ ] Implement Stage1-like framewise AR inference.
- [ ] Implement or verify framewise causal self-attention semantics.
- [ ] Implement full-video fallback path if self KV cache is disabled.
- [ ] Implement self KV cache history path.
- [ ] Split CFG self KV cache into cond/uncond caches.
- [ ] Fix Tokamax/Splash block-size behavior for growing KV length.
- [ ] Add focused tests for AR helper behavior.
- [ ] Add focused tests for framewise causal attention semantics.
- [ ] Add focused tests for Tokamax block-size selection.
- [ ] Run target TPU command to completion.
- [ ] Locate generated mp4.
- [ ] Inspect generated mp4 quality.
- [ ] If quality is poor, debug reproduction mismatch until video is clear and usable.

## Debug Notes

Activate environment:

```bash
source /home/bennett121358/maxdiffusion_venv/bin/activate
cd /home/bennett121358/maxdiffusion
```

If TPU lockfile blocks local tests:

```bash
JAX_PLATFORMS=cpu python -m pytest <test_file>
```

For any full generation failure, record:

- exact command
- final traceback
- generated output path, if any
- whether output video exists
- visual quality notes
- checkpoint load logs
- loaded parameter count or key mismatch logs
- `flash_block_sizes`
- frame index or cache length if visible
- whether `framewise_ar_use_self_kv_cache` is enabled

## Codex Goal Usage

推荐 goal 描述：

```text
创建一个 goal：在 causal_attention 分支复现 Causal Forcing Stage 1 的 Wan2.1 T2V AR diffusion 结构和功能，使用 /home/bennett121358/ar_diffusion.pt 跑通目标命令并生成清晰可用的视频。
```

继续执行时可以说：

```text
继续当前 goal，先检查当前代码和官方 Stage 1 结构还差什么。
```

或者：

```text
继续当前 goal，检查最新 TPU run 的报错或生成质量，并修复直到视频可用。
```

如果目标完成，可以标记 complete；如果连续多轮因为同一个外部条件无法推进，例如 TPU 不可用或 checkpoint 无法访问，可以标记 blocked。
