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
  hvp = HVP(calc_loss=criterion,
            parameters=[p for p in trained_model.parameters() if p.requires_grad],
            data=train_dataloader,
            num_batches=len(train_dataloader),
            damping=0.01)
  cg = CG(matmul=hvp,
          result_len=len(hvp.parameters),
          max_iters=len(hvp.parameters))
  test_hvps = []
  for x_test, label in test_dataset:
    x_test_batch = [torch.tensor(arg).unsqueeze(0) for arg in x_test]
    label_batch = torch.tensor([label])
    loss_at_x_test = criterion(trained_model(*x_test_batch), label_batch)
    grad_at_z_test = autograd.grad(loss_at_x_test, trained_model.parameters())
    test_hvps.append(cg.solve(grad_at_z_test))
  return torch.stack(test_hvps)


def calc_influence(criterion: Callable,
                   trained_model: nn.Module,
                   train_sample: Tuple[torch.Tensor, torch.Tensor],
                   test_hvps: torch.Tensor):
  features, target = train_sample
  train_loss = criterion(trained_model(features), target)
  params = trained_model.parameters()
  grad_at_train_sample = autograd.grad(train_loss, params)
  return test_hvps.matmul(grad_at_train_sample)
