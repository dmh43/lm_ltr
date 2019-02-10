from typing import Callable, Tuple, Any, Optional, Mapping, List

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset

from fastai.basic_data import DeviceDataLoader
from fastai import to_device
from progressbar import progressbar

from .hvp import HVP
from .gnp import GNP
from .cg import CG
from .grad_helpers import collect
from .utils import maybe

def _calc_laplacian(): raise NotImplementedError()

def calc_test_hvps(criterion: Callable,
                   trained_model: nn.Module,
                   train_dataloader: DeviceDataLoader,
                   test_dataloader: DataLoader,
                   run_params: Mapping,
                   diff_wrt: Optional[torch.Tensor]=None,
                   show_progress: bool=False):
  device = train_dataloader.device
  diff_wrt = maybe(diff_wrt, [p for p in trained_model.parameters() if p.requires_grad])
  if run_params['use_gauss_newton']:
    matmul_class = GNP
    damping = 0.001
  else:
    matmul_class = HVP
    damping = 0.01
  matmul = matmul_class(calc_loss=lambda xs, target: criterion(trained_model(*xs), target),
                        parameters=diff_wrt,
                        data=train_dataloader,
                        data_len=len(train_dataloader),
                        damping=damping,
                        cache_batch=True)
  cg = CG(matmul=matmul,
          result_len=sum(p.numel() for p in diff_wrt),
          max_iters=maybe(run_params['max_cg_iters'],
                          sum(p.numel() for p in diff_wrt)))
  test_hvps: List[torch.Tensor] = []
  tmp = test_dataloader.dataset.use_weighted_loss
  if getattr(run_params, 'weight_influence', False):
    test_dataloader.dataset.use_weighted_loss = True
  else:
    test_dataloader.dataset.use_weighted_loss = False
  iterator = progressbar(test_dataloader) if show_progress else test_dataloader
  for batch in iterator:
    x_test, target = to_device(batch, device)
    label = target > 0
    loss_at_x_test = criterion(trained_model(*x_test), label.squeeze())
    grads = autograd.grad(loss_at_x_test, diff_wrt)
    grad_at_z_test = collect(grads)
    hvp_weight = abs(target).sum() / len(target)
    test_hvps.append(hvp_weight * cg.solve(grad_at_z_test,
                                           test_hvps[-1] if len(test_hvps) != 0 else None))
    matmul.clear_batch()
  test_dataloader.dataset.use_weighted_loss = tmp
  return torch.stack(test_hvps)


def calc_influence(criterion: Callable,
                   trained_model: nn.Module,
                   train_sample: Tuple[torch.Tensor, torch.Tensor],
                   test_hvps: torch.Tensor,
                   diff_wrt: Optional[torch.Tensor]=None):
  features, target = train_sample
  train_loss = criterion(trained_model(*features), target)
  diff_wrt = maybe(diff_wrt, [p for p in trained_model.parameters() if p.requires_grad])
  grads = autograd.grad(train_loss, diff_wrt)
  grad_at_train_sample = collect(grads)
  with torch.no_grad():
    influence = test_hvps.matmul(grad_at_train_sample)
  return influence

def get_num_neg_influences(criterion: Callable,
                           trained_model: nn.Module,
                           train_sample: Tuple[torch.Tensor, torch.Tensor],
                           test_hvps: torch.Tensor,
                           thresh: Optional[float]=0.0,
                           diff_wrt: Optional[torch.Tensor]=None):
  influence = calc_influence(criterion, trained_model, train_sample, test_hvps, diff_wrt=diff_wrt)
  return torch.sum(influence < thresh)
