import torch
import torch.nn as nn

import lm_ltr.hvp as hvp

def test_hvp_1():
  loss = nn.MSELoss(reduce=False)
  out = torch.tensor(torch.arange(-5, 5, 0.1), requires_grad=True)
  targets = (torch.arange(-5, 5, 0.1) * 3)
  calc_hvp = hvp.HVP(lambda x,t: loss(x,t).sum(), out, zip(out, targets), num_batches=len(out))
  assert all(calc_hvp(torch.tensor([0.5])) == 1 / len(out))

def test_hvp_quadratic():
  loss = nn.MSELoss(reduce=False)
  out = torch.tensor(torch.arange(-5, 5, 0.1), requires_grad=True)
  targets = (torch.arange(-5, 5, 0.1) * 3)
  calc_hvp = hvp.HVP(lambda x,t: (loss(x,t) ** 2).sum(), out, zip(out, targets), num_batches=len(out))
  assert (calc_hvp(torch.tensor([0.5])) - 6 * (out - targets) ** 2 / len(out)).norm() < 1e-5
