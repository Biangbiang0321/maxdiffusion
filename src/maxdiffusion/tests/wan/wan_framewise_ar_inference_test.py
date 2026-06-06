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

import jax.numpy as jnp

from maxdiffusion.pipelines.wan.wan_framewise_ar_utils import (
    append_self_kv_cache,
    flowmatch_euler_step,
    make_flowmatch_schedule,
    make_framewise_ar_timestep,
    replace_latent_frame_block,
)


def test_make_framewise_ar_timestep_uses_batch_and_frame_block_shape():
  timestep = jnp.array(123, dtype=jnp.int32)

  result = make_framewise_ar_timestep(timestep, batch_size=2, num_frames=3)

  assert result.shape == (2, 3)
  assert result.dtype == jnp.int32
  assert (result == 123).all()


def test_replace_latent_frame_block_updates_only_target_block():
  latents = jnp.zeros((1, 2, 5, 1, 1), dtype=jnp.float32)
  block = jnp.ones((1, 2, 2, 1, 1), dtype=jnp.float32)

  result = replace_latent_frame_block(latents, block, frame_start=2, num_frames=2)

  assert (result[:, :, :2] == 0).all()
  assert (result[:, :, 2:4] == 1).all()
  assert (result[:, :, 4:] == 0).all()


def test_append_self_kv_cache_appends_sequence_axis():
  first_key = jnp.ones((2, 1, 3, 4, 5), dtype=jnp.bfloat16)
  first_value = jnp.ones((2, 1, 3, 4, 5), dtype=jnp.bfloat16) * 2
  second_key = jnp.ones((2, 1, 3, 2, 5), dtype=jnp.bfloat16) * 3
  second_value = jnp.ones((2, 1, 3, 2, 5), dtype=jnp.bfloat16) * 4

  cache = append_self_kv_cache(None, (first_key, first_value))
  cache = append_self_kv_cache(cache, (second_key, second_value))

  key, value = cache
  assert key.shape == (2, 1, 3, 6, 5)
  assert value.shape == (2, 1, 3, 6, 5)
  assert (key[:, :, :, :4] == 1).all()
  assert (key[:, :, :, 4:] == 3).all()
  assert (value[:, :, :, :4] == 2).all()
  assert (value[:, :, :, 4:] == 4).all()


def test_flowmatch_schedule_and_euler_step_match_official_last_step_semantics():
  timesteps, sigmas = make_flowmatch_schedule(num_inference_steps=2, shift=1.0, num_train_timesteps=1000)

  assert timesteps.tolist() == [1000.0, 500.0]
  assert sigmas.tolist() == [1.0, 0.5]

  sample = jnp.ones((1, 1), dtype=jnp.float32)
  model_output = jnp.ones((1, 1), dtype=jnp.float32) * 2

  first = flowmatch_euler_step(model_output, sample, sigmas, step_index=0)
  second = flowmatch_euler_step(model_output, first, sigmas, step_index=1)

  assert jnp.allclose(first, 0.0)
  assert jnp.allclose(second, -1.0)
