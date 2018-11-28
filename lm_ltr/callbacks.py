from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn

from fastai.basic_train import LearnerCallback

@dataclass
class ClampPositive(LearnerCallback):
  ps: List[nn.Parameter]

  def on_backward_end(self, **kwargs):
    for p in self.ps:
      p.clamp_(min=0)
