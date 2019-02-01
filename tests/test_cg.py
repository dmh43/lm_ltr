import torch

import lm_ltr.cg as cg

def test_cg():
  matmul = lambda vec: torch.eye(2).matmul(vec)
  solver = cg.CG(matmul, 2)
  assert all(solver.solve(torch.ones(2)) == torch.ones(2))

def test_ident_scale():
  matmul = lambda vec: torch.eye(2).matmul(vec) * 2
  solver = cg.CG(matmul, 2)
  assert all(solver.solve(torch.ones(2)) == torch.ones(2) / 2)

def test_cg_std():
  mat = torch.tensor([[1.0, 2.0], [1.0, 4.0]])
  matmul = lambda vec: mat.matmul(vec)
  preconditioner = cg.get_preconditioner(lambda: torch.diag(mat))
  solver = cg.CG(matmul, 2, max_iters=1000, preconditioner=preconditioner)
  assert all(solver.solve(torch.ones(2)) == mat.inverse().matmul(torch.ones(2)))
