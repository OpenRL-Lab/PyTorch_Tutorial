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

import time

import torch

def time_evaluation(origin, compiled, input, exec_func=None, exp_name: str ='',warmup_time: int =1) -> None:
    torch.cuda.synchronize()
    s_t = time.time()
    exec_func(origin,input) if exec_func else origin(input)
    torch.cuda.synchronize()
    start_t1 = time.time() - s_t
    print(f"Normal firstly used time:{start_t1}s")

    torch.cuda.synchronize()
    s_t = time.time()
    exec_func(compiled,input) if exec_func else compiled(input)
    torch.cuda.synchronize()
    start_t2 = time.time() - s_t
    print(f"Compiled firstly used time:{start_t2}s")

    assert warmup_time>=1
    for _ in range(warmup_time-1):
        exec_func(compiled, input) if exec_func else compiled(input)


    t_1_total, t_2_total = 0., 0.
    for i in range(10):
        torch.cuda.synchronize()
        s_t = time.time()
        exec_func(origin,input) if exec_func else origin(input)
        torch.cuda.synchronize()
        t_1 = time.time() - s_t
        t_1_total += t_1

        torch.cuda.synchronize()
        s_t = time.time()
        exec_func(compiled,input) if exec_func else compiled(input)
        torch.cuda.synchronize()
        t_2 = time.time() - s_t
        t_2_total += t_2

        print(f"{i}:\n\tNormal used time:{t_1}s, \n\t"
              f"Compiled used time:{t_2}s")

    print(f"{exp_name}在编译前的首次运行时间为:{start_t1}秒")
    print(f"{exp_name}在编译后的首次运行时间为:{start_t2}秒")
    print(f"{exp_name}在后续运行过程中的加速比为:{t_1_total / t_2_total:.2f}")
