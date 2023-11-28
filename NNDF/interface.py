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
FRAMEWORK_NATIVE = "native"
FRAMEWORK_TENSORRT = "trt"
FRAMEWORK_ONNXRT = "onnxrt"
VALID_FRAMEWORKS = [
    FRAMEWORK_NATIVE,
    FRAMEWORK_ONNXRT,
    FRAMEWORK_TENSORRT
]

class MetadataArgparseInteropMixin:
    """Add argparse support where the class can add new arguments to an argparse object."""

    @staticmethod
    @abstractmethod
    def add_args(parser):
        pass

    @staticmethod
    @abstractmethod
    def from_args(args):
        pass

    @staticmethod
    @abstractmethod
    def add_inference_args(parser):
        pass

    @staticmethod
    @abstractmethod
    def from_inference_args(args):
        pass

    @staticmethod
    @abstractmethod
    def add_benchmarking_args(parser):
        """
        Add args needed for perf benchmarking mode.
        """
        pass

