from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import scatter_kwargs

class TupleDataParallel(nn.DataParallel):
  def forward(self, *inputs: Tuple[torch.Tensor]):
    if not self.device_ids:
      return self.module(*inputs)
    inputs = scatter_kwargs(inputs, None, self.device_ids, self.dim)
    if len(self.device_ids) == 1:
      return self.module(*inputs[0])
    replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
    outputs = self.parallel_apply(replicas, inputs, None)
    return self.gather(outputs, self.output_device)
