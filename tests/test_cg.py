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

def test_cg_ill_conditioned():
  mat = torch.tensor([[1.0, 0.0], [1.0, 40.0]])
  matmul = lambda vec: mat.matmul(vec)
  preconditioner = cg.get_preconditioner(lambda: torch.diag(mat))
  solver_with_preconditioner = cg.CG(matmul, 2, max_iters=2, preconditioner=preconditioner)
  solver = cg.CG(matmul, 2, max_iters=2)
  no_precondition_err = solver.solve(torch.ones(2)) - mat.inverse().matmul(torch.ones(2))
  with_precondition_err = solver_with_preconditioner.solve(torch.ones(2)) - mat.inverse().matmul(torch.ones(2))
  assert no_precondition_err.norm() > with_precondition_err.norm()
  assert all(abs(with_precondition_err) < 0.1)

def test_cg_ill_conditioned_many_iters():
  mat = torch.tensor([[1.0, 0.0], [1.0, 40.0]])
  matmul = lambda vec: mat.matmul(vec)
  preconditioner = cg.get_preconditioner(lambda: torch.diag(mat))
  solver = cg.CG(matmul, 2, max_iters=20, preconditioner=preconditioner)
  with_precondition_err = solver.solve(torch.ones(2)) - mat.inverse().matmul(torch.ones(2))
  assert all(abs(with_precondition_err) < 1e-5)

def test_cg_ill_conditioned_many_iters_no_pre():
  mat = torch.tensor([[1.0, 0.0], [1.0, 40.0]])
  matmul = lambda vec: mat.matmul(vec)
  solver = cg.CG(matmul, 2, max_iters=20)
  with_precondition_err = solver.solve(torch.ones(2)) - mat.inverse().matmul(torch.ones(2))
  assert all(abs(with_precondition_err) > 1e-5)
