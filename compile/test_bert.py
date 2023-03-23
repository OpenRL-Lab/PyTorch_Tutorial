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
from transformers import BertTokenizer, BertModel

from utils import time_evaluation

# 从huggingface上加载未训练的BERT模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").to(device="cuda:0")

# 编译优化模型
compiled_model = torch.compile(model)

# 准备输入数据
text = "Hello World!"
encoded_input = tokenizer(text, return_tensors='pt').to(device="cuda:0")

def exec_fun(model,input):
    model(**input)

# 测试
time_evaluation(model,compiled_model,encoded_input,exec_fun,'BERT')