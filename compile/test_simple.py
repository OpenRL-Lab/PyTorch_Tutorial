#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The TARTRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""

# 简单测试

import torch

from utils import time_evaluation


# 一个简单的函数
def simple_fn(x):
    for _ in range(20):
        y = torch.sin(x).cuda()
        x = x + y
    return x


compiled_fn = torch.compile(simple_fn, backend="inductor")

input_tensor = torch.randn(10000).to(device="cuda:0")

# 测试
time_evaluation(simple_fn, compiled_fn, input_tensor, None, '简单函数')
