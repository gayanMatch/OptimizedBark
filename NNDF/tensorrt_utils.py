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

import tensorrt as trt
import torch
from functools import reduce
from NNDF.networks import NetworkMetadata
from NNDF.models import TRTEngineFile
from NNDF.logger import G_LOGGER


def allocate_binding_buffer(types_dict, shapes_dict):
    '''
    Allocate binding buffers for trt based on provided types and shapes dict
    '''
    return {
        k: torch.zeros(reduce(lambda v, a: v*a, shape), dtype=types_dict[k]).cuda()
        for k, shape in shapes_dict.items()
    }


# Helper Classes
class TRTNativeRunner:
    """TRTNativeRunner avoids the high overheads with Polygraphy runner providing performance comparable to C++ implementation."""
    def __init__(self, trt_engine_file: TRTEngineFile, network_metadata: NetworkMetadata):
        self.network_metadata = network_metadata
        self.trt_engine_file = trt_engine_file
        self.trt_logger = trt.Logger()

        if G_LOGGER.level == G_LOGGER.DEBUG:
            self.trt_logger.min_severity = trt.Logger.VERBOSE
        elif G_LOGGER.level == G_LOGGER.INFO:
            self.trt_logger.min_severity = trt.Logger.INFO
        else:
            self.trt_logger.min_severity = trt.Logger.WARNING

        G_LOGGER.info("Reading and loading engine file {} using trt native runner.".format(self.trt_engine_file.fpath))
        with open(self.trt_engine_file.fpath, "rb") as f:
            self.trt_runtime = trt.Runtime(self.trt_logger)
            self.trt_engine = self.trt_runtime.deserialize_cuda_engine(f.read())
            self.trt_context = self.trt_engine.create_execution_context()

        # By default set optimization profile to 0
        self.profile_idx = 0

        # Other metadata required by the profile
        self._num_bindings_per_profile = self.trt_engine.num_bindings // self.trt_engine.num_optimization_profiles
        G_LOGGER.debug("Number of profiles detected in engine: {}".format(self._num_bindings_per_profile))

    def release(self):
        pass

    def get_optimization_profile(self, batch_size, sequence_length):
        """Provided helper function to obtain a profile optimization."""
        # Select an optimization profile
        # inspired by demo/BERT/inference.py script
        selected_profile_idx = None
        for idx in range(self.trt_engine.num_optimization_profiles):
            profile_shape = self.trt_engine.get_profile_shape(profile_index=idx, binding=idx * self._num_bindings_per_profile)

            if profile_shape[0][0] <= batch_size and profile_shape[2][0] >= batch_size \
               and profile_shape[0][1] <=  sequence_length and profile_shape[2][1] >= sequence_length:
                G_LOGGER.debug("Selected profile: {}".format(profile_shape))
                selected_profile_idx = idx
                break

        if selected_profile_idx == -1:
            raise RuntimeError("Could not find any profile that matches batch_size={}, sequence_length={}".format(batch_size, sequence_length))

        return selected_profile_idx

    def __call__(self, *args, **kwargs):
        self.trt_context.active_optimization_profile = self.profile_idx
        return self.forward(*args, **kwargs)
