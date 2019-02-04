import torch
import torch.nn as nn

import lm_ltr.hvp as hvp

def test_hvp():
  loss = nn.MSELoss()
  param = torch.tensor(1.0, requires_grad=True)
  x = torch.arange(-5, 5, 0.1)
  out = x * param
  scale = 3
  targets = x * scale
  calc_hvp = hvp.HVP(loss, [param], zip(out, targets), data_len=len(out))
  vec = torch.tensor([0.5])
  assert abs(calc_hvp(vec) - torch.sum((scale - 1) * x ** 2 * vec) / len(out)) < 1e-3
