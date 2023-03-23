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

import torch
from torchvision import models

from utils import time_evaluation

model = models.resnet50().cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
compiled_model = torch.compile(model)

input = torch.randn(16, 3, 224, 224).cuda()

def exec_func(model, input):
    # 执行训练程序
    optimizer.zero_grad()
    out = model(input)
    out.sum().backward()
    optimizer.step()

# 测试
time_evaluation(model,compiled_model, input, exec_func,"resnet50")
