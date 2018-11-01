from random import shuffle
from functools import partial

from fastai.metrics import accuracy_thresh
from fastai import fit, GradientClipping, Learner

import torch
from torch.optim import Adam
import torch.nn as nn
import pydash as _
import torch.nn.functional as F

from .metrics import RankingMetricRecorder, recall, precision, f1
from .losses import hinge_loss
from .recorders import PlottingRecorder, LossesRecorder

def _get_loss_function(use_pointwise_loss, regularize=None):
  def _with_regularization(loss_fn, regularize):
    def _calc_reg(regularize):
      loss = []
      for reg_type, param in regularize:
        if reg_type == 'l2':
          loss.append(torch.sum(param ** 2))
        else:
          raise NotImplementedError('Only implemented for l2, not for: ' + reg_type)
      return sum(loss)
    def _apply_loss_fn(out, target):
      return loss_fn(out, target) + _calc_reg(regularize)
    return _apply_loss_fn
  if use_pointwise_loss:
    loss_fn = F.mse_loss
  else:
    loss_fn = hinge_loss
  return _with_regularization(loss_fn, regularize) if regularize is not None else loss_fn

def train_model(model,
                model_data,
                train_ranking_dataset,
                test_ranking_dataset,
                train_params,
                model_params,
                experiment,
                regularize=None):
  model = nn.DataParallel(model)
  loss = _get_loss_function(train_params.use_pointwise_loss, regularize=regularize)
  metrics = []
  callbacks = [RankingMetricRecorder(model_data.device,
                                     model.module.pointwise_scorer if hasattr(model.module, 'pointwise_scorer') else model,
                                     train_ranking_dataset,
                                     test_ranking_dataset,
                                     experiment,
                                     doc_chunk_size=train_params.batch_size if model_params.use_pretrained_doc_encoder else -1)]
  if train_params.use_gradient_clipping:
    callback_fns = [partial(GradientClipping, clip=train_params.gradient_clipping_norm),
                    partial(PlottingRecorder, experiment),
                    partial(LossesRecorder, experiment)]
  else:
    callback_fns=[]
  print("Training:")
  learner = Learner(model_data,
                    model,
                    opt_fn=Adam,
                    loss_fn=loss,
                    metrics=metrics,
                    callbacks=callbacks,
                    callback_fns=callback_fns,
                    wd=train_params.weight_decay)
  learner.fit(train_params.num_epochs)
  torch.save(model.state_dict(), './model_save_' + experiment.model_name)
