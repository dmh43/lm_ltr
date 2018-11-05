import torch
import torch.nn as nn
import torch.nn.functional as F

from toolz import cons

from .losses import hinge_loss

class MultiObjective(nn.Module):
  def __init__(self, model, train_params, additive=None):
    super().__init__()
    self.add_rel_score = train_params.add_rel_score
    self.use_pointwise_loss = train_params.add_rel_score
    self.rel_score_penalty = train_params.rel_score_penalty
    self.rel_score_obj_scale = train_params.rel_score_obj_scale
    self.additive = additive
    self.model = model

  def loss(self, multi_objective_out, target):
    if self.use_pointwise_loss:
      loss_fn = F.mse_loss
    else:
      loss_fn = hinge_loss
    pred_out = multi_objective_out[0]
    pred_loss = loss_fn(pred_out, target)
    side_loss = sum(multi_objective_out[1:])
    reg = sum([p ** 2 for p in self.additive.parameters()])
    return pred_loss + self.rel_score_obj_scale * side_loss + self.rel_score_penalty * reg

  def _pointwise_forward(self, query, document, lens):
    rel_score = self.rel_score(query, document)
    out = self.model(query, document, lens)
    return (out, rel_score)

  def _pairwise_forward(self, query, document_1, document_2, lens_1, lens_2):
    rel_score_1 = self.rel_score(query, document_1)
    rel_score_2 = self.rel_score(query, document_2)
    out = self.model(query, document_1, document_2, lens_1, lens_2)
    return (out, rel_score_1, rel_score_2)

  def forward(self, *args):
    if self.use_pointwise_loss:
      query, document, lens = args
      return self._pointwise_forward(query, document, lens)
    else:
      query, document_1, document_2, lens_1, lens_2 = args
      return self._pairwise_forward(query, document_1, document_2, lens_1, lens_2)
