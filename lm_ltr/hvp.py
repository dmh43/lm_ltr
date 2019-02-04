from dataclasses import dataclass
from typing import Callable, Optional, Iterable, Tuple, List, Any, Iterator

import torch

from .grad_helpers import collect
from .utils import maybe

@dataclass
class HVP:
  calc_loss: Callable
  parameters: List[torch.Tensor]
  data: Iterable[Tuple[torch.Tensor, torch.Tensor]]
  data_len: int
  damping: float=0.0
  grad_vec: Optional[torch.Tensor] = None
  device: Optional[torch.device] = None
  cache_batch: bool = False
  _batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

  def __post_init__(self, *args, **kwargs):
    self._data_iter = iter(self.data)

  def _zero_grad(self):
    for p in self.parameters:
      if p.grad is not None:
        p.grad.data.zero_()

  def clear_batch(self): self._batch = None

  def _get_batch(self):
    try:
      self._batch = next(self._data_iter)
    except StopIteration:
      self._data_iter = iter(self.data)
      self._batch = next(self._data_iter)
    return self._batch

  def _batch_forward(self, hessian_vec_prod, x_chunk, target, vec):
    loss = self.calc_loss(x_chunk, target)
    grad_dict = torch.autograd.grad(loss, self.parameters, create_graph=True)
    grad_vec = collect(grad_dict)
    grad_product = grad_vec.dot(vec)
    grad_grad = torch.autograd.grad(grad_product, self.parameters, retain_graph=True)
    if self.cache_batch:
      hessian_vec_prod += collect(grad_grad)
    else:
      hessian_vec_prod += collect(grad_grad) / self.data_len

  def __call__(self, vec) -> torch.Tensor:
    """
    Returns H*vec where H is the hessian of the loss w.r.t.
    the vectorized model parameters
    """
    num_iters = 0
    hessian_vec_prod = torch.zeros_like(vec)
    hessian_vec_prod.requires_grad = False
    self._zero_grad()
    if self.cache_batch:
      x_chunk, target = maybe(self._batch, self._get_batch())
      self._batch_forward(hessian_vec_prod, x_chunk, target, vec)
    else:
      for x_chunk, target in self.data:
        self._batch_forward(hessian_vec_prod, x_chunk, target, vec)
        num_iters += 1
    return hessian_vec_prod + self.damping * vec
