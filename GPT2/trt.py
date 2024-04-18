#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pycuda.driver as cuda
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig
from NNDF.models import TRTEngineFile
from NNDF.tensorrt_utils import TRTNativeRunner


class GPT2TRTDecoder(TRTNativeRunner):
    def __init__(
            self,
            trt_engine_file: TRTEngineFile,
            variant: str,
            hf_config: PretrainedConfig,
            batch_size: int = 1,
    ):
        super().__init__(trt_engine_file)
        self.variant = variant
        self.config = hf_config
        self.batch_size = batch_size
        self.data_type = torch.float16
        # In benchmarking mode, if input_profile_max is provided, should use that as max_sequence_length
        self.max_sequence_length = GPT2ModelTRTConfig.MAX_LENGTH[variant]

        # Similarly, the max_output_length should be the user-provided output_profile_max_len if provided
        self.max_output_length = self.max_sequence_length

        self.main_input_name = "input_ids"
        self.num_heads = self.config.n_head
        self.embedding_size_per_head = self.config.n_embd // self.num_heads
        self.num_decoder_layers = self.config.n_layer

        self.profile_idx = 0
        self.bindings = [0] * self.trt_engine.num_bindings
        self.logits = torch.zeros((self.batch_size, 1, hf_config.vocab_size), dtype=self.data_type).cuda()

        self.bindings[self.trt_engine.get_binding_index("logits")] = self.logits.data_ptr()

        # Setting input and output the same does not work for GPT2.
        # Needs separate cache and copy the memory address after each iteration
        self.self_attention_cache_1 = {}
        self.self_attention_cache_2 = {}

        self_attention_kv_shape = (
            self.batch_size, self.num_heads, self.max_output_length - 1, self.embedding_size_per_head)

        # Set kv cache shape and type
        for i in range(self.num_decoder_layers):
            for code in ["key", "value"]:
                self_attention_name = f"key_values.{i}.decoder.{code}"
                kv_buffer_1 = torch.zeros(self_attention_kv_shape, dtype=self.data_type).cuda()
                kv_buffer_2 = torch.zeros(self_attention_kv_shape, dtype=self.data_type).cuda()
                self.self_attention_cache_1[self_attention_name] = kv_buffer_1
                self.self_attention_cache_2[self_attention_name] = kv_buffer_2

                input_idx = self.trt_engine.get_binding_index("past_" + self_attention_name)
                output_idx = self.trt_engine.get_binding_index("present_" + self_attention_name)

                self.bindings[input_idx] = kv_buffer_1.data_ptr()  # Generation phase
                self.bindings[output_idx] = kv_buffer_2.data_ptr()

        self.kv_cache_binding_offset = 1  # 0: input_ids, kv cache input indices start from 1
        self.past_decoder_length = 0
        self.use_cache_1_as_input = True

        self.context_mode = self.config.use_cache
        self.return_device = torch.device('cuda')
        self.device = torch.device('cuda')

    def _switch_input_output_binding(self):
        '''
        For kv cache mode, switch input and output pointers to avoid data concurrency issue and D2D copy
        '''
        # When context mode (output in cache 1) and cache 1 is used as inputs, no need to switch bindings
        if not (self.use_cache_1_as_input and self.context_mode):
            for i in range(self.num_decoder_layers):
                for code in ["key", "value"]:
                    self_attention_name = f"key_values.{i}.decoder.{code}"
                    input_idx = self.trt_engine.get_binding_index("past_" + self_attention_name)
                    output_idx = self.trt_engine.get_binding_index("present_" + self_attention_name)

                    # Switch generation mode kv cache bindings
                    temp = self.bindings[output_idx]
                    self.bindings[output_idx] = self.bindings[input_idx]
                    self.bindings[input_idx] = temp
            self.use_cache_1_as_input = not self.use_cache_1_as_input

    def load_past_key_values(self, past_key_values):
        for i in range(self.num_decoder_layers):
            cuda.memcpy_htod(self.bindings[self.trt_engine.get_binding_index(f"past_key_values.{i}.decoder.key")],
                             past_key_values[i][0].contiguous().cpu().numpy())
            cuda.memcpy_htod(self.bindings[self.trt_engine.get_binding_index(f"past_key_values.{i}.decoder.value")],
                             past_key_values[i][1].contiguous().cpu().numpy())
        self.past_decoder_length = past_key_values[0][0].shape[2]

    def forward(self, input_ids, *args, **kwargs):
        bs = input_ids.shape[0]
        input_length = input_ids.shape[1]

        # Check if the input data is on CPU (which usually means the PyTorch does not support current GPU).
        is_cpu_mode = (input_ids.device == torch.device("cpu")) or (self.return_device == "cpu")

        if is_cpu_mode:
            input_ids = input_ids.cuda()

        # Set the binding shape of input_ids, which should be (bs, input_length).
        self.bindings[0] = input_ids.data_ptr()
        self.trt_context.set_binding_shape(0, input_ids.shape)

        self_attention_kv_shape = (bs, self.num_heads, self.past_decoder_length, self.embedding_size_per_head)

        for i in range(self.num_decoder_layers):
            self.trt_context.set_binding_shape(self.kv_cache_binding_offset + 2 * i, self_attention_kv_shape)
            self.trt_context.set_binding_shape(self.kv_cache_binding_offset + 2 * i + 1, self_attention_kv_shape)

        # Launch TRT inference.
        assert self.trt_context.all_binding_shapes_specified
        self.trt_context.execute_v2(bindings=self.bindings)

        # For bs > 1, this is required, so cannnot avoid this D2D copy
        logits_length = bs * 1 * self.config.vocab_size
        logits = self.logits.flatten()[:logits_length].view(bs, 1, self.config.vocab_size)

        if is_cpu_mode:
            logits = logits.cpu()

        present_key_values = None
        self.past_decoder_length += input_length

        present_key_values = ()
        self_attention_cache = self.self_attention_cache_1 if self.use_cache_1_as_input or (
                self.profile_idx == 0) else self.self_attention_cache_2

        for i in range(self.num_decoder_layers):

            self_attention_k_output = self_attention_cache[f"key_values.{i}.decoder.key"]
            self_attention_v_output = self_attention_cache[f"key_values.{i}.decoder.value"]

            if is_cpu_mode:
                self_attention_k_output = self_attention_k_output.cpu()
                self_attention_v_output = self_attention_v_output.cpu()

            present_key_values += ((self_attention_k_output, self_attention_v_output),)
        self._switch_input_output_binding()
        return CausalLMOutputWithPast(logits=logits.to(self.return_device), past_key_values=present_key_values)
