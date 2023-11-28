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
from itertools import tee
import tensorrt as trt
from NNDF.models import TRTEngineFile
TEXT_ENCODING_OFFSET = 10048
TEXT_PAD_TOKEN = 129_595
SEMANTIC_PAD_TOKEN = 10_000
SEMANTIC_INFER_TOKEN = 129_599


# TRT Engine File Encoding #
class GPT2TRTEngine(TRTEngineFile):
    def __init__(self, model, network_metadata):
        super().__init__(model, network_metadata)

    def use_obey_precision_constraints(self):
        return self.network_metadata.precision.fp16

    def get_network_definition(self, network_definition):

        def pairwise(iterable):
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)

        indices = list(range(0, network_definition[1].num_layers))
        for i, i_next in pairwise(indices):
            l = network_definition[1].get_layer(i)
            l_next = network_definition[1].get_layer(i_next)

            if not all([l.get_output(i).is_execution_tensor for i in range(l.num_outputs)]):
                continue

            if l.get_output_type(0) != trt.float32:
                continue

            if l.type == trt.LayerType.ELEMENTWISE and l_next.type == trt.LayerType.REDUCE:
                l.__class__ = getattr(trt, "IElementWiseLayer")
                if l.op == trt.ElementWiseOperation.POW:
                    l.precision = trt.float32
                    l.set_output_type(0, trt.float32)

                l_next.precision = trt.float32
                l_next.set_output_type(0, trt.float32)

        if self.network_metadata.precision.fp16:
            for i in range(network_definition[1].num_inputs):
                t = network_definition[1].get_input(i)
                if t.dtype == trt.float32:
                    t.dtype = trt.float16

            for i in range(network_definition[1].num_outputs):
                t = network_definition[1].get_output(i)
                if t.dtype == trt.float32:
                    t.dtype = trt.float16

        return network_definition
