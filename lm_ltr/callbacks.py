from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn

from fastai.basic_train import Callback, LearnerCallback

class MaxIter(Callback):
  def __init__(self, max_iter, *args, **kwargs):
    self.max_iter = max_iter
    super().__init__(*args, **kwargs)

  def on_batch_end(self, num_batch):
    if self.max_iter is not None and num_batch >= self.max_iter: return True

@dataclass
class ClampPositive(LearnerCallback):
  ps: List[nn.Parameter]

  def on_backward_end(self, **kwargs):
    for p in self.ps:
      p.data.clamp_(min=0)
