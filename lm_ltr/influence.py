from typing import Callable, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset

from fastai.basic_data import DeviceDataLoader
from fastai import to_device

from .hvp import HVP
from .cg import CG

def _calc_laplacian(): raise NotImplementedError()

def calc_test_hvps(criterion: Callable,
                   trained_model: nn.Module,
                   train_dataloader: DeviceDataLoader,
                   test_dataset: Dataset,
                   collate_fn: Callable[[Any], torch.Tensor]):
  device = train_dataloader.device
  diff_wrt = [p for p in trained_model.parameters() if p.requires_grad]
  hvp = HVP(calc_loss=lambda xs, target: criterion(trained_model(*xs), target),
            parameters=diff_wrt,
            data=train_dataloader,
            num_batches=len(train_dataloader),
            damping=0.01)
  cg = CG(matmul=hvp,
          result_len=sum(p.numel() for p in diff_wrt),
          max_iters=sum(p.numel() for p in diff_wrt))
  test_hvps = []
  for sample in test_dataset:
    x_test, label = to_device(collate_fn([sample]), device)
    loss_at_x_test = criterion(trained_model(*x_test), label.squeeze())
    grads = autograd.grad(loss_at_x_test, diff_wrt)
    grad_at_z_test = torch.cat([g.contiguous().view(-1) for g in grads])
    test_hvps.append(cg.solve(grad_at_z_test))
  return torch.stack(test_hvps)


def calc_influence(criterion: Callable,
                   trained_model: nn.Module,
                   train_sample: Tuple[torch.Tensor, torch.Tensor],
                   test_hvps: torch.Tensor):
  features, target = train_sample
  train_loss = criterion(trained_model(*features), target)
  diff_wrt = [p for p in trained_model.parameters() if p.requires_grad]
  grads = autograd.grad(train_loss, diff_wrt)
  grad_at_train_sample = torch.cat([g.contiguous().view(-1) for g in grads])
  with torch.no_grad():
    influence = test_hvps.matmul(grad_at_train_sample)
  return influence
