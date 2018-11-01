from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import scatter_kwargs

class TupleDataParallel(nn.DataParallel):
  """Expects that a batch is in the form Tuple[Tuple[Tensor], Tensor]"""

  def forward(self, *inputs: Tuple[Tuple[torch.Tensor], torch.Tensor]):
    if not self.device_ids:
      return self.module(*inputs)
    inputs = self.scatter(inputs, self.device_ids)
    if len(self.device_ids) == 1:
      return self.module(*inputs[0])
    replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
    outputs = self.parallel_apply(replicas, inputs, None)
    return self.gather(outputs, self.output_device)

  def scatter(self, inputs, device_ids):
    y_scattered = scatter_kwargs(inputs[-1], None, device_ids, self.dim)
    x_scattered = list(zip([scatter_kwargs(x, None, device_ids, self.dim) for x in inputs[:-1]]))
    return list(zip(x_scattered, y_scattered))
