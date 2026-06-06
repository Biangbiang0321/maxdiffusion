# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


SelfKvCache = Optional[Tuple[jax.Array, jax.Array]]


def make_framewise_ar_timestep(timestep: jax.Array, batch_size: int, num_frames: int) -> jax.Array:
  """Builds the official Stage 1 style timestep tensor, shaped [B, current_frames]."""
  return jnp.broadcast_to(timestep, (batch_size, num_frames))


def reset_scheduler_state_for_ar_block(scheduler, scheduler_state, num_inference_steps: int, block_shape: Tuple[int, ...]):
  """Resets UniPC scheduler history so each AR block has an independent denoise trajectory."""
  return scheduler.set_timesteps(
      scheduler_state,
      num_inference_steps=num_inference_steps,
      shape=block_shape,
  )


def make_flowmatch_schedule(
    num_inference_steps: int,
    shift: float,
    num_train_timesteps: int = 1000,
) -> Tuple[jax.Array, jax.Array]:
  """Builds the official Causal Forcing FlowMatch schedule.

  Official Stage 1 uses FlowMatchScheduler(extra_one_step=True, sigma_min=0).
  """
  sigmas = jnp.linspace(1.0, 0.0, num_inference_steps + 1, dtype=jnp.float32)[:-1]
  sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
  timesteps = sigmas * num_train_timesteps
  return timesteps, sigmas


def flowmatch_euler_step(model_output: jax.Array, sample: jax.Array, sigmas: jax.Array, step_index: int) -> jax.Array:
  sigma = sigmas[step_index]
  sigma_next = jnp.where(step_index + 1 >= sigmas.shape[0], jnp.array(0.0, dtype=sigmas.dtype), sigmas[step_index + 1])
  return sample + model_output * (sigma_next - sigma)


def replace_latent_frame_block(
    latents: jax.Array,
    block_latents: jax.Array,
    frame_start: int,
    num_frames: int,
) -> jax.Array:
  """Writes a generated latent frame block into a full latent video tensor."""
  return latents.at[:, :, frame_start : frame_start + num_frames].set(block_latents[:, :, :num_frames])


def append_self_kv_cache(self_kv_cache: SelfKvCache, present_self_kv: Tuple[jax.Array, jax.Array]) -> SelfKvCache:
  """Appends one clean AR block of self-attention K/V to the history cache.

  Shape convention is [layers, batch, heads, sequence, head_dim].
  """
  present_key, present_value = jax.tree.map(jax.lax.stop_gradient, present_self_kv)
  if self_kv_cache is None:
    return present_key, present_value

  cached_key, cached_value = self_kv_cache
  return (
      jnp.concatenate([cached_key, present_key], axis=3),
      jnp.concatenate([cached_value, present_value], axis=3),
  )
