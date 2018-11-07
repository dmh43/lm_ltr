import torch
import torch.nn as nn
import torch.nn.functional as F

from toolz import cons, partial

from .losses import hinge_loss

class MultiObjective(nn.Module):
  def __init__(self, model, train_params, rel_score=None, additive=None):
    super().__init__()
    self.add_rel_score = train_params.add_rel_score
    self.use_pointwise_loss = train_params.use_pointwise_loss
    self.rel_score_penalty = train_params.rel_score_penalty
    self.rel_score_obj_scale = train_params.rel_score_obj_scale
    self.margin = train_params.margin
    self.additive = additive
    self.rel_score = rel_score
    self.model = model

  def loss(self, multi_objective_out, target):
    if self.use_pointwise_loss:
      loss_fn = F.mse_loss
    else:
      loss_fn = partial(hinge_loss, margin=self.margin)
    pred_out = multi_objective_out[0]
    pred_loss = loss_fn(pred_out, target)
    if self.add_rel_score:
      side_loss = torch.sum(multi_objective_out[1:])
      reg = torch.sum([p ** 2 for p in self.additive.parameters()])
      return pred_loss + self.rel_score_obj_scale * side_loss + self.rel_score_penalty * reg
    else:
      return pred_loss

  def _pointwise_forward(self, query, document, lens):
    if self.add_rel_score:
      rel_score = self.rel_score(query, document) if self.add_rel_score else 0
      out = self.model(query, document, lens)
      return (out, rel_score)
    else:
      return (out,)

  def _pairwise_forward(self, query, document_1, document_2, lens_1, lens_2):
    if self.add_rel_score:
      rel_score_1 = self.rel_score(query, document_1) if self.add_rel_score else 0
      rel_score_2 = self.rel_score(query, document_2) if self.add_rel_score else 0
      out = self.model(query, document_1, document_2, lens_1, lens_2)
      return (out, rel_score_1, rel_score_2)
    else:
      out = self.model(query, document_1, document_2, lens_1, lens_2)
      return (out,)

  def forward(self, *args):
    if self.use_pointwise_loss:
      query, document, lens = args
      return self._pointwise_forward(query, document, lens)
    else:
      query, document_1, document_2, lens_1, lens_2 = args
      return self._pairwise_forward(query, document_1, document_2, lens_1, lens_2)
