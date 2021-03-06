import torch
import torch.nn as nn
import torch.nn.functional as F

from toolz import cons, partial

from .losses import hinge_loss, truncated_hinge_loss, bce_loss, smoothed_bce_loss, l1_loss

class MultiObjective(nn.Module):
  def __init__(self, model, train_params, rel_score=None, additive=None):
    super().__init__()
    self.add_rel_score = train_params.add_rel_score
    self.use_pointwise_loss = train_params.use_pointwise_loss
    self.use_truncated_hinge_loss = train_params.use_truncated_hinge_loss
    self.use_variable_loss = train_params.use_variable_loss
    self.use_weighted_loss = train_params.use_weighted_loss
    self.use_bce_loss = train_params.use_bce_loss
    self.use_label_smoothing = train_params.use_label_smoothing
    self.use_l1_loss = train_params.use_l1_loss
    self.use_noise_aware_loss = train_params.use_noise_aware_loss
    self.truncation = train_params.truncation
    self.rel_score_penalty = train_params.rel_score_penalty
    self.rel_score_obj_scale = train_params.rel_score_obj_scale
    self.margin = train_params.margin
    self.additive = additive
    self.rel_score = rel_score
    self.model = model
    if self.use_pointwise_loss:
      self.loss_fn = F.mse_loss
    elif self.use_truncated_hinge_loss:
      self.loss_fn = partial(truncated_hinge_loss,
                             margin=self.margin,
                             truncation=self.truncation)
    elif self.use_noise_aware_loss:
      self.loss_fn = nn.BCEWithLogitsLoss()
    elif self.use_variable_loss:
      self.loss_fn = hinge_loss
    elif self.use_bce_loss:
      self.loss_fn = bce_loss
    elif self.use_label_smoothing:
      self.loss_fn = smoothed_bce_loss
    elif self.use_l1_loss:
      self.loss_fn = l1_loss
    else:
      self.loss_fn = partial(hinge_loss,
                             margin=self.margin)

  def loss(self, multi_objective_out, target, **kwargs):
    pred_out = multi_objective_out[0]
    if self.use_variable_loss:
      margin = torch.abs(target)
      rounded_target = (target > 0).float() - (target < 0).float()
      pred_loss = self.loss_fn(pred_out, rounded_target, margin, **kwargs)
    elif self.use_weighted_loss:
      weight = torch.abs(target)
      rounded_target = (target > 0).float() - (target < 0).float()
      pred_loss = self.loss_fn(pred_out, rounded_target, weight, **kwargs)
    else:
      pred_loss = self.loss_fn(pred_out, target, **kwargs)
    if self.add_rel_score:
      if getattr(kwargs, 'reduction', None) is 'none': raise NotImplementedError
      side_loss = torch.sum(sum(multi_objective_out[1:]))
      reg = torch.sum(sum([p ** 2 for p in self.additive.parameters()]))
      return pred_loss + self.rel_score_obj_scale * side_loss + self.rel_score_penalty * reg
    else:
      return pred_loss

  def _pointwise_forward(self, query, document, lens, doc_score, **kwargs):
    if self.add_rel_score:
      rel_score = self.rel_score(query, document) if self.add_rel_score else 0
      out = self.model(query, document, lens, doc_score, **kwargs)
      return (out, rel_score)
    else:
      out = self.model(query, document, lens, doc_score, **kwargs)
      return (out,)

  def _pairwise_forward(self, query, document_1, document_2, lens_1, lens_2, doc_1_scores, doc_2_scores, **kwargs):
    if self.add_rel_score:
      rel_score_1 = self.rel_score(query, document_1) if self.add_rel_score else 0
      rel_score_2 = self.rel_score(query, document_2) if self.add_rel_score else 0
      out = self.model(query, document_1, document_2, lens_1, lens_2, doc_1_scores, doc_2_scores, **kwargs)
      return (out, rel_score_1, rel_score_2)
    else:
      out = self.model(query, document_1, document_2, lens_1, lens_2, doc_1_scores, doc_2_scores, **kwargs)
      return (out,)

  def forward(self, *args, **kwargs):
    if self.use_pointwise_loss:
      return self._pointwise_forward(*args, **kwargs)
    else:
      return self._pairwise_forward(*args, **kwargs)
