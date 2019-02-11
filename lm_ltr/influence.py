from typing import Callable, Tuple, Any, Optional, Mapping, List, Union, Type

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset
from scipy.optimize import fmin_ncg
import numpy as np

from fastai.basic_data import DeviceDataLoader
from fastai import to_device
from progressbar import progressbar
import pydash as _

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
    matmul_class: Union[Type[GNP], Type[HVP]] = GNP
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
  iterator = progressbar(test_dataloader) if show_progress else test_dataloader
  for batch in iterator:
    x_test, target = to_device(batch, device)
    loss_at_x_test = criterion(trained_model(*x_test), target.squeeze())
    grads = autograd.grad(loss_at_x_test, diff_wrt)
    grad_at_z_test = collect(grads)
    if len(test_hvps) != 0:
      init = test_hvps[-1].detach().clone()
    else:
      init = torch.zeros_like(grad_at_z_test)
    def _min(x, grad_at_z_test=grad_at_z_test):
      x_tens = torch.tensor(x)
      grad_tens = torch.tensor(grad_at_z_test)
      return np.array(0.5 * matmul(x_tens).dot(x_tens) - grad_tens.dot(x_tens))
    def _grad(x, grad_at_z_test=grad_at_z_test):
      x_tens = torch.tensor(x)
      return np.array(matmul(x_tens) - grad_at_z_test)
    def _hess(x, p):
      grad_tens = torch.tensor(p)
      return np.array(matmul(grad_tens))
    if getattr(run_params, 'use_scipy', False):
      test_hvps.append(torch.tensor(fmin_ncg(f=_min,
                                             x0=init,
                                             fprime=_grad,
                                             fhess_p=_hess,
                                             avextol=1e-8,
                                             maxiter=100,
                                             disp=False),
                                    device=grad_at_z_test.device))
    else:
      test_hvps.append(cg.solve(grad_at_z_test,
                                test_hvps[-1] if len(test_hvps) != 0 else None))
    matmul.clear_batch()
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

def calc_dataset_influence(trained_model: nn.Module,
                           to_last_layer: Callable,
                           train_dataloader_sequential: DeviceDataLoader,
                           test_hvps: torch.Tensor,
                           sum_p: bool=False):
  with torch.no_grad():
    influences = []
    for x, target in train_dataloader_sequential:
      plus_minus_target = 2 * target -1
      neg_like = - torch.sigmoid(- trained_model(*x)[0] * plus_minus_target)
      features = to_last_layer(x)[0]
      bias = torch.ones(len(features), dtype=features.dtype, device=features.device)
      in_last_layer = torch.cat([features, bias.unsqueeze(1)], 1)
      grads_at_train_batch = in_last_layer * neg_like.unsqueeze(1) * plus_minus_target.unsqueeze(1)
      result = grads_at_train_batch.matmul(test_hvps.t())
      influences.append(result.sum(1) if sum_p else result)
  return torch.cat(influences, 0)

def get_num_neg_influences(criterion: Callable,
                           trained_model: nn.Module,
                           train_sample: Tuple[torch.Tensor, torch.Tensor],
                           test_hvps: torch.Tensor,
                           thresh: Optional[float]=0.0,
                           diff_wrt: Optional[torch.Tensor]=None):
  influence = calc_influence(criterion, trained_model, train_sample, test_hvps, diff_wrt=diff_wrt)
  return torch.sum(influence < thresh)
