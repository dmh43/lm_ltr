from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset

from .hvp import HVP
from .cg import CG, get_preconditioner

def _calc_laplacian(): raise NotImplementedError()

def calc_test_hvps(criterion: Callable,
                   trained_model: nn.Module,
                   train_dataloader: DataLoader,
                   test_dataset: Dataset):
  hvp = HVP(calc_loss=lambda xs, target: criterion(trained_model(xs), target),
            parameters=[p for p in trained_model.parameters() if p.requires_grad],
            data=train_dataloader,
            num_batches=len(train_dataloader),
            damping=0.01)
  cg = CG(matmul=hvp,
          result_len=sum(p.numel() for p in hvp.parameters),
          max_iters=sum(p.numel() for p in hvp.parameters))
  test_hvps = []
  for x_test, label in test_dataset:
    if isinstance(x_test, tuple): x_test_batch = [torch.tensor(arg).unsqueeze(0) for arg in x_test]
    else                        : x_test_batch = x_test.unsqueeze(0)
    if isinstance(x_test_batch, tuple): loss_at_x_test = criterion(trained_model(*x_test_batch), label)
    else:                               loss_at_x_test = criterion(trained_model(x_test_batch), label)
    grads = autograd.grad(loss_at_x_test, trained_model.parameters())
    grad_at_z_test = torch.cat([g.contiguous().view(-1) for g in grads])
    test_hvps.append(cg.solve(grad_at_z_test))
  return torch.stack(test_hvps)


def calc_influence(criterion: Callable,
                   trained_model: nn.Module,
                   train_sample: Tuple[torch.Tensor, torch.Tensor],
                   test_hvps: torch.Tensor):
  features, target = train_sample
  train_loss = criterion(trained_model(features), target)
  params = trained_model.parameters()
  grads = autograd.grad(train_loss, params)
  grad_at_train_sample = torch.cat([g.contiguous().view(-1) for g in grads])
  return test_hvps.matmul(grad_at_train_sample)
