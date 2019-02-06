from typing import Callable, Tuple, Any, Optional, Mapping

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset

from fastai.basic_data import DeviceDataLoader
from fastai import to_device

from .hvp import HVP
from .cg import CG
from .grad_helpers import collect
from .utils import maybe

def _calc_laplacian(): raise NotImplementedError()

def calc_test_hvps(criterion: Callable,
                   trained_model: nn.Module,
                   train_dataloader: DeviceDataLoader,
                   test_dataloader: DataLoader,
                   run_params: Mapping):
  device = train_dataloader.device
  diff_wrt = [p for p in trained_model.parameters() if p.requires_grad]
  hvp = HVP(calc_loss=lambda xs, target: criterion(trained_model(*xs), target),
            parameters=diff_wrt,
            data=train_dataloader,
            data_len=len(train_dataloader),
            damping=0.01,
            cache_batch=True)
  cg = CG(matmul=hvp,
          result_len=sum(p.numel() for p in diff_wrt),
          max_iters=maybe(run_params['max_cg_iters'],
                          sum(p.numel() for p in diff_wrt)))
  test_hvps = []
  for batch in test_dataloader:
    x_test, label = to_device(batch, device)
    loss_at_x_test = criterion(trained_model(*x_test), label.squeeze())
    grads = autograd.grad(loss_at_x_test, diff_wrt)
    grad_at_z_test = collect(grads)
    test_hvps.append(cg.solve(grad_at_z_test,
                              test_hvps[-1] if len(test_hvps) != 0 else None))
    hvp.clear_batch()
  return torch.stack(test_hvps)


def calc_influence(criterion: Callable,
                   trained_model: nn.Module,
                   train_sample: Tuple[torch.Tensor, torch.Tensor],
                   test_hvps: torch.Tensor):
  features, target = train_sample
  train_loss = criterion(trained_model(*features), target)
  diff_wrt = [p for p in trained_model.parameters() if p.requires_grad]
  grads = autograd.grad(train_loss, diff_wrt)
  grad_at_train_sample = collect(grads)
  with torch.no_grad():
    influence = test_hvps.matmul(grad_at_train_sample)
  return influence

def get_num_neg_influences(criterion: Callable,
                           trained_model: nn.Module,
                           train_sample: Tuple[torch.Tensor, torch.Tensor],
                           test_hvps: torch.Tensor,
                           thresh: Optional[float]=0.0):
  influence = calc_influence(criterion, trained_model, train_sample, test_hvps)
  return torch.sum(influence < thresh)
