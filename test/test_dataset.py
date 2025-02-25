# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2025/2/14 11:04
# @File: test_dataset
# @Email: mlshenkai@163.com
from tensordict.tensordict import TensorDict
import torch
a = TensorDict({"a": torch.rand(3), "b": torch.rand(3,4)}, [3])



print(a)
