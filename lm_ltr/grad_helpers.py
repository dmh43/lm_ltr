from typing import List

import torch

def collect(grads: List[torch.Tensor]) -> torch.Tensor:
  coll = []
  for g in grads:
    coll.append(g.contiguous().view(-1))
  return torch.cat(coll)
