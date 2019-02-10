from dataclasses import dataclass
from typing import Callable, Optional, Iterable, Tuple, List, Any, Iterator

import torch

from .grad_helpers import collect
from .utils import maybe

@dataclass
class GNP:
  """Gauss-Newton Product. See http://andrew.gibiansky.com/blog/machine-learning/gauss-newton-matrix/"""
  calc_loss: Callable
  parameters: List[torch.Tensor]
  data: Iterable[Tuple[torch.Tensor, torch.Tensor]]
  data_len: int
  damping: float=0.0
  device: Optional[torch.device] = None
  cache_batch: bool = False

  def __post_init__(self, *args, **kwargs):
    self._batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    self._data_iter = iter(self.data)
    self._grad_vec: Optional[torch.Tensor] = None

  def _zero_grad(self):
    for p in self.parameters:
      if p.grad is not None:
        p.grad.data.zero_()

  def clear_batch(self):
    self._batch = None
    self._grad_vec = None

  def _get_batch(self):
    try:
      self._batch = next(self._data_iter)
    except StopIteration:
      self._data_iter = iter(self.data)
      self._batch = next(self._data_iter)
    return self._batch

  def _batch_forward(self, gn_vec_prod, x_chunk, target, vec):
    if self.cache_batch and self._grad_vec is not None:
      grad_vec = self._grad_vec
    else:
      loss = self.calc_loss(x_chunk, target)
      grad_dict = torch.autograd.grad(loss, self.parameters, create_graph=True)
      grad_vec = collect(grad_dict)
    if self.cache_batch and self._grad_vec is None:
      self._grad_vec = grad_vec
    grad_product = grad_vec.dot(vec)
    if self.cache_batch:
      gn_vec_prod += grad_product * grad_vec
    else:
      gn_vec_prod += grad_product * grad_vec / self.data_len

  def __call__(self, vec) -> torch.Tensor:
    """
    Returns G*vec where G is the Gauss-Newton approximation to the
    hessian of the loss w.r.t.  the vectorized model parameters
    """
    num_iters = 0
    gn_vec_prod = torch.zeros_like(vec)
    gn_vec_prod.requires_grad = False
    self._zero_grad()
    if self.cache_batch:
      x_chunk, target = self._batch if self._batch is not None else self._get_batch()
      self._batch_forward(gn_vec_prod, x_chunk, target, vec)
    else:
      for x_chunk, target in self.data:
        self._batch_forward(gn_vec_prod, x_chunk, target, vec)
        num_iters += 1
    return gn_vec_prod + self.damping * vec
