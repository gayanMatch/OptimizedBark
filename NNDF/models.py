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
from abc import abstractmethod
from NNDF.networks import NetworkMetadata


class TRTEngineFile:

    @abstractmethod
    def use_obey_precision_constraints(self):
        pass

    # get_network_definition can be overloaded to alter the network definition.
    # For example, this function can be used to change the precisions of ops or
    # data type of intermediate tensors.
    def get_network_definition(self, network_definition):
        return network_definition

    def __init__(
        self,
        model: str,
        network_metadata: NetworkMetadata = None,
    ):
        self.network_metadata = network_metadata
        self.fpath = model
        self.max_trt_workspace = 3072
