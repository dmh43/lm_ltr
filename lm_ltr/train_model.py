from random import shuffle
from functools import partial

from fastai.metrics import accuracy_thresh
from fastai import fit, GradientClipping, Learner
from fastai.train import lr_find, AdamW

import torch
from torch.optim import Adam, SGD, RMSprop
import torch.nn as nn
import pydash as _
import torch.nn.functional as F

import matplotlib.pyplot as plt

from .metrics import RankingMetricRecorder, recall, precision, f1
from .losses import hinge_loss
from .recorders import PlottingRecorder, LossesRecorder
from .inits import weight_init
from .callbacks import MaxIter

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
                valid_ranking_dataset,
                test_ranking_dataset,
                train_params,
                model_params,
                run_params,
                experiment):
  model.apply(weight_init)
  loss = model.loss
  model = nn.DataParallel(model)
  metrics = []
  callbacks = [RankingMetricRecorder(model_data.device,
                                     _get_pointwise_scorer(model),
                                     train_ranking_dataset,
                                     valid_ranking_dataset,
                                     test_ranking_dataset,
                                     experiment,
                                     doc_chunk_size=train_params.batch_size if model_params.use_pretrained_doc_encoder else -1,
                                     dont_smooth=model_params.dont_smooth,
                                     dont_include_normalized_score=model_params.dont_include_normalized_score,
                                     use_dense=model_params.use_dense,
                                     record_every_n=run_params.record_every_n),
               MaxIter(train_params.max_iter)]
  callback_fns = []
  if train_params.use_gradient_clipping:
    callback_fns.append(partial(GradientClipping, clip=train_params.gradient_clipping_norm))
  # other_recorders = partial(PlottingRecorder, experiment),
  callback_fns.extend([partial(LossesRecorder, experiment)])
  print("Training:")
  if train_params.optimizer == 'adam':
    opt_func = Adam
  elif train_params.optimizer == 'adamw':
    opt_func = AdamW
  elif train_params.optimizer == 'rmsprop':
    opt_func = RMSprop
  elif train_params.optimizer == 'sgd':
    opt_func = partial(SGD, lr=train_params.learning_rate)
  learner = Learner(model_data,
                    model,
                    opt_func=opt_func,
                    loss_func=loss,
                    metrics=metrics,
                    callbacks=callbacks,
                    callback_fns=callback_fns,
                    wd=train_params.weight_decay)
  if train_params.use_cyclical_lr:
    lr_find(learner, num_it=1000)
    learner.recorder.plot()
    plt.savefig('./lr.png')
  else:
    learner.fit(train_params.num_epochs, lr=train_params.learning_rate)
  torch.save(model.module.state_dict(), './model_save_' + experiment.model_name)
