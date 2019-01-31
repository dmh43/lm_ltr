from typing import Callable, Optional

import torch

class HVP:
  def __init__(self,
               calc_loss: Callable,
               parameters: torch.Tensor,
               xs: torch.Tensor,
               targets: torch.Tensor,
               batch_size: int=512):
    self.calc_loss = calc_loss
    self.xs = xs
    self.targets = targets
    self.parameters = parameters
    self.batch_size = batch_size
    self.grad_vec: Optional[torch.Tensor] = None

  def _zero_grad(self):
    for p in self.parameters:
      if p.grad is not None:
        p.grad.data.zero_()

  def _grad(self) -> torch.Tensor:
    num_chunks = max(1, len(self.xs) // self.batch_size)
    grad_vec = None
    for x_chunk, target in zip(self.xs.chunk(num_chunks),
                               self.targets.chunk(num_chunks)):
      loss = self.calc_loss(x_chunk, target)
      grad_dict = torch.autograd.grad(loss, self.parameters, create_graph=True)
      grad_vec_chunk = torch.cat([g.contiguous().view(-1) for g in grad_dict])
      grad_vec = grad_vec + grad_vec_chunk if grad_vec is not None else grad_vec_chunk
    grad_vec /= num_chunks
    self.grad_vec = grad_vec
    return grad_vec

  def __call__(self, vec) -> torch.Tensor:
    """
    Returns H*vec where H is the hessian of the loss w.r.t.
    the vectorized model parameters
    """
    self._zero_grad()
    grad_vec = self.grad_vec if self.grad_vec is not None else self._grad()
    grad_product = torch.sum(grad_vec * vec)
    self._zero_grad()
    grad_grad = torch.autograd.grad(grad_product, self.parameters)
    hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in grad_grad])
    return hessian_vec_prod
