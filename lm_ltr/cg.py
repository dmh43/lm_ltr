from typing import Optional, Callable
from dataclasses import dataclass

import torch

from .utils import maybe

def get_preconditioner(diag: torch.Tensor, cond_offset: Optional[float]=0.01):
  """Take the diagonal as an approximation to the matrix. Obviously a
     pretty low quality preconditioner. Usually works better than not
     preconditioning if the matrix is illconditioned and PSD
  """
  return 1.0 / (diag + cond_offset)

@dataclass
class CG:
  """Solves for result in the system given by: matmul(result) = vec"""
  matmul: Callable
  result_len: int
  tol: float = 1e-8
  max_iters: Optional[int] = None
  preconditioner: Optional[torch.Tensor] = None
  compute_res_exactly_lim: int = 50

  def _check_max_iters(self, num_iters):
    return (self.max_iters is None) or (num_iters < self.max_iters)

  def _apply_preconditioner(self, vec: torch.Tensor):
    return self.preconditioner * vec if self.preconditioner is not None else vec

  def solve(self, vec: torch.Tensor, init: Optional[torch.Tensor]=None):
    """See `An Introduction to the Conjugate Gradient Method Without the
       Agonizing Pain` by Jonathan Richard Shewchuk"""
    result = maybe(init, torch.zeros(self.result_len, device=vec.device, dtype=vec.dtype))
    r = vec - self.matmul(result)
    d = self._apply_preconditioner(r)
    delta_new = r.dot(d)
    delta_init = delta_new
    num_iters = 0
    while delta_new > self.tol ** 2 * delta_init and self._check_max_iters(num_iters):
      q = self.matmul(d)
      step_size = delta_new / d.dot(q)
      result += step_size * d
      if num_iters % self.compute_res_exactly_lim == 0:
        r = vec - self.matmul(result)
      else:
        r -= step_size * q
      s = self._apply_preconditioner(r)
      delta_old = delta_new
      delta_new = r.dot(s)
      d_step_size = delta_new / delta_old
      d = s + d_step_size * d
      num_iters += 1
    return result
