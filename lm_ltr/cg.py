from typing import Optional, Callable
from dataclasses import dataclass

import torch

def get_preconditioner(get_diag: Callable, cond_offset: Optional[float]=0.01):
  diag = get_diag()
  smallest_elem = torch.min(diag)
  offset = cond_offset + smallest_elem if smallest_elem < 0 else cond_offset
  return 1.0 / (diag + offset)

@dataclass
class CG:
  """Solves for result in the system given by: matmul(result) = vec"""
  matmul: Callable
  result_len: int
  tol: float = 1e-8
  max_iters: Optional[int] = None
  preconditioner: Optional[torch.Tensor] = None

  def _check_max_iters(self, num_iters):
    return (self.max_iters is None) or (num_iters < self.max_iters)

  def _apply_preconditioner(self, vec: torch.Tensor):
    return self.preconditioner * vec if self.preconditioner is not None else vec

  def solve(self, vec: torch.Tensor):
    result = torch.zeros(self.result_len, device=vec.device)
    err = self.matmul(result) - self._apply_preconditioner(vec)
    p = - err
    num_iters = 0
    while any(abs(err) > self.tol) and self._check_max_iters(num_iters):
      matmul_p = self._apply_preconditioner(self.matmul(p))
      den = p.dot(matmul_p)
      err_norm = err.dot(err)
      step_size = err_norm / den
      result += step_size * p
      err += step_size * matmul_p
      err_step_size = err.dot(err) / err_norm
      p = - err + err_step_size * p
      num_iters += 1
    return result
