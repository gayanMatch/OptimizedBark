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

# TRT-HuggingFace

class GPT2ModelTRTConfig:
    NUMBER_OF_LAYERS = {
        'bark': 12,
        'bark_large': 24,
        'bark_no_cache': 12,
        'bark_coarse': 12,
        'bark_coarse_large': 24,
        'bark_coarse_no_cache': 12
    }
    MAX_LENGTH = {
        'bark': 512 + 768,
        'bark_large': 512 + 768,
        'bark_no_cache': 512 + 768,
        'bark_coarse': 946,
        'bark_coarse_large': 946,
        'bark_coarse_no_cache': 946
    }

    MIN_OUTPUT_LENGTH = {
        'bark': 0,
        'bark_large': 0,
        'bark_no_cache': 0,
        'bark_coarse': 0,
        'bark_coarse_large': 0,
        'bark_coarse_no_cache': 0
    }
