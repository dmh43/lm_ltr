import torch
import torch.nn as nn

import lm_ltr.hvp as hvp

def test_hvp_zero():
  loss = nn.MSELoss(reduce=False)
  out = torch.tensor(torch.arange(-5, 5, 0.1), requires_grad=True)
  targets = (torch.arange(-5, 5, 0.1))
  calc_hvp = hvp.HVP(lambda x,t: loss(x,t).sum(), out, out, targets)
  assert all(calc_hvp(torch.tensor([0.5])) == 0)

def test_hvp_1():
  loss = nn.MSELoss(reduce=False)
  out = torch.tensor(torch.arange(-5, 5, 0.1), requires_grad=True)
  targets = (torch.arange(-5, 5, 0.1) * 3)
  calc_hvp = hvp.HVP(lambda x,t: loss(x,t).sum(), out, out, targets)
  assert all(calc_hvp(torch.tensor([0.5])) == 1)

def test_hvp_quadratic():
  loss = nn.MSELoss(reduce=False)
  out = torch.tensor(torch.arange(-5, 5, 0.1), requires_grad=True)
  targets = (torch.arange(-5, 5, 0.1) * 3)
  calc_hvp = hvp.HVP(lambda x,t: (loss(x,t) ** 2).sum(), out, out, targets)
  assert all(calc_hvp(torch.tensor([0.5])) == 6 * (out - targets) ** 2)
