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
from .callbacks import ClampPositive

def _get_pointwise_scorer(model):
  if hasattr(model.module.model, 'pointwise_scorer'):
    return model.module.model.pointwise_scorer
  else:
    return model.module.model

def _get_term_weights_params(model):
  pointwise_scorer = _get_pointwise_scorer(model)
  params = []
  params.extend(pointwise_scorer.document_encoder.weights.parameters())
  params.extend(pointwise_scorer.query_encoder.weights.parameters())
  return params

def train_model(model,
                model_data,
                train_ranking_dataset,
                test_ranking_dataset,
                train_params,
                model_params,
                experiment):
  loss = model.loss
  model = nn.DataParallel(model)
  metrics = []
  callbacks = [RankingMetricRecorder(model_data.device,
                                     _get_pointwise_scorer(model),
                                     train_ranking_dataset,
                                     test_ranking_dataset,
                                     experiment,
                                     doc_chunk_size=train_params.batch_size if model_params.use_pretrained_doc_encoder else -1)]
  callback_fns = []
  if train_params.use_gradient_clipping:
    callback_fns.append(partial(GradientClipping, clip=train_params.gradient_clipping_norm))
  callback_fns.extend([partial(PlottingRecorder, experiment),
                       partial(LossesRecorder, experiment),
                       partial(ClampPositive, ps=_get_term_weights_params(model))])
  print("Training:")
  learner = Learner(model_data,
                    model,
                    opt_func=Adam,
                    loss_func=loss,
                    metrics=metrics,
                    callbacks=callbacks,
                    callback_fns=callback_fns,
                    wd=train_params.weight_decay)
  learner.fit(train_params.num_epochs, lr=train_params.learning_rate)
  torch.save(model.state_dict(), './model_save_' + experiment.model_name)
