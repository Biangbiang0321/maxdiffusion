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

from collections import OrderedDict

import torch

from maxdiffusion.models.wan.wan_utils import (
    _extract_causal_forcing_generator_state_dict,
    rename_for_custom_trasformer,
)


def test_extract_causal_forcing_generator_state_dict_prefers_generator_key():
  generator_state = OrderedDict([("model.patch_embedding.weight", torch.zeros((1, 1, 1, 1, 1)))])
  checkpoint = {"generator": generator_state, "other": {"unused": torch.ones((1,))}}

  assert _extract_causal_forcing_generator_state_dict(checkpoint) is generator_state


def test_rename_for_custom_transformer_strips_official_model_prefix():
  renamed = rename_for_custom_trasformer("model.blocks_0.self_attn.q.weight")

  assert renamed == "blocks.0.attn1.query.weight"


def test_rename_for_custom_transformer_maps_block_modulation_to_adaln():
  renamed = rename_for_custom_trasformer("model.blocks_0.modulation")

  assert renamed == "blocks.0.adaln_scale_shift_table"


def test_rename_for_custom_transformer_maps_official_embedding_names():
  renamed = rename_for_custom_trasformer("model.text_embedding_0.weight")

  assert renamed == "condition_embedder.text_embedder.linear_1.weight"
