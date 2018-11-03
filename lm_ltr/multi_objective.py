import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import hinge_loss

class MultiObjective(nn.Module):
  def __init__(self, model, side_objectives, regularization, use_pointwise_loss):
    side_models, side_loss = list(zip(*side_objectives))
    self.models = nn.ModuleList([model] + side_models)
    self.losses = [1.0] + side_loss
    self.regularization = regularization
    self.use_pointwise_loss = use_pointwise_loss

  def loss(self, multi_objective_out, target):
    if self.use_pointwise_loss:
      loss_fn = F.mse_loss
    else:
      loss_fn = hinge_loss
    pred_out = multi_objective_out[0]
    pred_loss = loss_fn(pred_out, target)
    side_loss = sum(multi_objective_out[1:])
    reg = sum([reg() for reg in self.regularization])
    return pred_out + side_loss + reg

  def forward(self, query, document):
    results = [loss * model(query, document) for model, loss in zip(self.models,
                                                                    self.losses)]
    return results
