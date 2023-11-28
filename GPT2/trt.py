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
import pycuda.autoinit
import torch
from typing import Dict, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin
from NNDF.networks import NetworkMetadata
from NNDF.tensorrt_utils import TRTNativeRunner, allocate_binding_buffer
from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig, GPT2BenchmarkingArgs
from NNDF.models import TRTEngineFile

class TRTHFRunner(TRTNativeRunner, GenerationMixin):
    """Runner that adds interop support for HF and HF provided greedy_search functions."""

    # Stores the encoder input length received at runtime, which is used to slice decoder inputs.
    ENCODER_LENGTH = 0
    def _allocate_memory(self,
                         input_shapes: Dict[str, tuple],
                         input_types: Dict[str, torch.dtype],
                         output_shapes: Dict[str, tuple],
                         output_types: Dict[str, torch.dtype]):
        """Helper function for binding several inputs at once and pre-allocating the results."""
        # Allocate memories as 1D linear buffers for simpler handling of dynamic shapes.
        self.inputs = allocate_binding_buffer(input_types, input_shapes)
        self.outputs = allocate_binding_buffer(output_types, output_shapes)

        bindings = [None] * self.trt_engine.num_bindings

        for input_name, input_array in self.inputs.items():
            # Allocate memory for inputs
            input_idx = self.trt_engine.get_binding_index(input_name)
            self.trt_context.set_binding_shape(input_idx, input_shapes[input_name])
            bindings[input_idx] = input_array.data_ptr()

        assert self.trt_context.all_binding_shapes_specified

        for output_name, output_array in self.outputs.items():
            # Output shape should be allocated from context size
            output_idx = self.trt_engine.get_binding_index(output_name)
            bindings[output_idx] = output_array.data_ptr()

        return bindings

    def __init__(
        self,
        trt_engine_file: TRTEngineFile,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
        batch_size: int = 1
    ):
        super().__init__(trt_engine_file, network_metadata)
        self.config = hf_config
        self.batch_size = batch_size

class GPT2TRTDecoder(TRTHFRunner):
    def __init__(
        self,
        trt_engine_file: str,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
        batch_size: int = 1,
        num_beams: int = 1,
        benchmarking_args: GPT2BenchmarkingArgs = None
    ):
        super().__init__(trt_engine_file, network_metadata, hf_config, batch_size = batch_size)
        self.network_metadata = network_metadata
        self.data_type = torch.float32 if not network_metadata.precision.fp16 else torch.float16
        # In benchmarking mode, if input_profile_max is provided, should use that as max_sequence_length
        if benchmarking_args is not None:
            if benchmarking_args.input_profile_max_len is not None:
                self.max_input_length = benchmarking_args.input_profile_max_len
            else:
                self.max_input_length = hf_config.n_positions
        # In non-benchmarking mode, we are provided a text generation task. We need to use the max_length as max sequence length
        else:
            self.max_sequence_length = GPT2ModelTRTConfig.MAX_LENGTH[network_metadata.variant]

        # Similarly, the max_output_length should be the user-provided output_profile_max_len if provided
        if benchmarking_args is not None and benchmarking_args.output_profile_max_len is not None:
            self.max_output_length = benchmarking_args.output_profile_max_len
        else:
            self.max_output_length = self.max_sequence_length

        self.main_input_name = "input_ids"
        self.num_heads = self.config.n_head
        self.embedding_size_per_head = self.config.n_embd // self.num_heads
        self.num_decoder_layers = self.config.n_layer

        self.profile_idx = 0
        self.bindings = [0] * self.trt_engine.num_bindings
        self.logits = torch.zeros((self.batch_size * num_beams, self.max_output_length, hf_config.vocab_size), dtype = self.data_type).cuda()
        self.bindings[self.trt_engine.get_binding_index("logits")] = self.logits.data_ptr()
        # This will be used to calculate the offset for each binding
        self.num_bindings = self.trt_engine.num_bindings // 2 if self.config.use_cache else self.trt_engine.num_bindings

        if self.config.use_cache:
            self.bindings[self.trt_engine.get_binding_index("logits") + self.num_bindings] = self.logits.data_ptr()
            
            # Setting input and output the same does not work for GPT2. Needs separate cache and copy the memory address after each iteration
            self.self_attention_cache_1 = {}
            self.self_attention_cache_2 = {}

            self_attention_kv_shape = (self.batch_size * num_beams, self.num_heads, self.max_output_length - 1, self.embedding_size_per_head)

            # Set kv cache shape and type
            for i in range(self.num_decoder_layers):
                for code in ["key", "value"]:

                    self_attention_name = f"key_values.{i}.decoder.{code}"
                    kv_buffer_1 = torch.ones(self_attention_kv_shape, dtype = self.data_type).cuda()
                    kv_buffer_2 = torch.zeros(self_attention_kv_shape, dtype = self.data_type).cuda()
                    self.self_attention_cache_1[self_attention_name] = kv_buffer_1
                    self.self_attention_cache_2[self_attention_name] = kv_buffer_2

                    input_idx = self.trt_engine.get_binding_index("past_" + self_attention_name)
                    output_idx = self.trt_engine.get_binding_index("present_" + self_attention_name)
                    
                    self.bindings[input_idx] = kv_buffer_1.data_ptr() # Generation phase
                    self.bindings[output_idx] = kv_buffer_2.data_ptr()  

                    # Context mode will always use buffer 1 as output
                    self.bindings[input_idx + self.num_bindings] = 0 # Context phase, should be 0
                    self.bindings[output_idx + self.num_bindings] = kv_buffer_1.data_ptr()

            self.kv_cache_binding_offset = 1 # 0: input_ids, kv cache input indices start from 1
            self.past_decoder_length = 0
            self.use_cache_1_as_input = True
            self._set_context_mode_trt_context()
        
        self.context_mode = self.config.use_cache
        self.return_device = torch.device('cuda')
        self.device = torch.device('cuda')

    def reset(self):
        '''
        Resets the input specific fields after finishing a task.
        '''
        self.context_mode = self.config.use_cache
    
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
 
    def prepare_inputs_for_generation(self, input_ids, past = None, use_cache = None, **kwargs):
        # TODO: add position_ids, token_type_ids support
        if past is not None:
            input_ids = input_ids[:, -1:]
            self.context_mode = False
        else:
            self.context_mode = self.config.use_cache
        
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": use_cache,
        }

    def set_return_device(self, return_device):
        """
        Sets the return device of the return via to(). Device name should be the same as torch devices: cuda, cpu, etc.
        This is used in our measurement code.
        """
        self.return_device = return_device

    def _reorder_cache(self, past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
    
    def _set_context_mode_trt_context(self):
        # Create TRT context for context mode (1st decoder run) with optimization profile = 1
        self.context_trt_context = self.trt_engine.create_execution_context()
        self.context_trt_context.active_optimization_profile = 1

    def load_past_key_values(self, past_key_values):
        for i in range(self.num_decoder_layers):
            cuda.memcpy_htod(self.bindings[self.trt_engine.get_binding_index(f"past_key_values.{i}.decoder.key")], past_key_values[i][0].contiguous().cpu().numpy())
            cuda.memcpy_htod(self.bindings[self.trt_engine.get_binding_index(f"past_key_values.{i}.decoder.value")], past_key_values[i][1].contiguous().cpu().numpy())
        self.past_decoder_length = past_key_values[0][0].shape[2]

    def forward(self, input_ids, *args, **kwargs):
        bs = input_ids.shape[0]
        input_length = input_ids.shape[1]

        # Check if the input data is on CPU (which usually means the PyTorch does not support current GPU).
        is_cpu_mode = (input_ids.device == torch.device("cpu")) or (self.return_device == "cpu")

        if is_cpu_mode:
            input_ids = input_ids.int().cuda()
        
        # Set the binding shape of input_ids, which should be (bs, input_length).
        if not self.context_mode:
            self.bindings[0] = input_ids.int().data_ptr()
            self.trt_context.set_binding_shape(0, input_ids.shape)
        else:
            self.bindings[self.num_bindings] = input_ids.int().data_ptr()
            self.context_trt_context.set_binding_shape(self.num_bindings, input_ids.shape)

        if self.config.use_cache:            
            if self.context_mode:
                self.past_decoder_length = 0

            self_attention_kv_shape = (bs, self.num_heads, self.past_decoder_length, self.embedding_size_per_head)

            for i in range(self.num_decoder_layers):
                if not self.context_mode:
                    # Optimization Profile 1 is generation phase with no kv inputs
                    self.trt_context.set_binding_shape(self.kv_cache_binding_offset+2*i, self_attention_kv_shape)
                    self.trt_context.set_binding_shape(self.kv_cache_binding_offset+2*i + 1, self_attention_kv_shape)
                else:
                    # Optimization Profile 0 is context phase with kv inputs
                    self.context_trt_context.set_binding_shape(self.kv_cache_binding_offset+2*i + self.num_bindings, self_attention_kv_shape)
                    self.context_trt_context.set_binding_shape(self.kv_cache_binding_offset+2*i + 1 + self.num_bindings, self_attention_kv_shape)
                    
        # Launch TRT inference.
        if not self.context_mode:
            assert self.trt_context.all_binding_shapes_specified
            self.trt_context.execute_v2(bindings=self.bindings)
        else:
            assert self.context_trt_context.all_binding_shapes_specified
            self.context_trt_context.execute_v2(bindings=self.bindings)
        
        # For bs > 1, this is required, so cannnot avoid this D2D copy
        logits_length = bs * 1 * self.config.vocab_size
        logits = self.logits.flatten()[:logits_length].view(bs, 1, self.config.vocab_size)

        if is_cpu_mode:
            logits = logits.cpu()

        present_key_values = None
        if self.config.use_cache:
            self.past_decoder_length += input_length

            present_key_values = ()
            self_attention_cache = self.self_attention_cache_1 if self.use_cache_1_as_input or (self.profile_idx == 0) else self.self_attention_cache_2
            
            for i in range(self.num_decoder_layers):

                self_attention_k_output = self_attention_cache[f"key_values.{i}.decoder.key"]
                self_attention_v_output = self_attention_cache[f"key_values.{i}.decoder.value"]

                if is_cpu_mode:
                    self_attention_k_output = self_attention_k_output.cpu()
                    self_attention_v_output = self_attention_v_output.cpu()

                present_key_values += ((self_attention_k_output, self_attention_v_output),) 

            self._switch_input_output_binding()
        return CausalLMOutputWithPast(logits=logits.to(self.return_device), past_key_values = present_key_values)
