from typing import List

import torch
import torch.nn as nn

class Regularization(nn.Module):
  def __init__(self, kind, module):
    super().__init__()
    self.kind = kind
    self.module = module

  def _l2_loss(self):
    return sum([p ** 2 for p in self.module.params()])

  def forward(self, *args, **kwargs):
    if self.kind == 'l2':
      return self._l2_loss()
    else:
      raise NotImplementedError('Only L2 reg supported')
