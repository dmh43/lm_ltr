from random import shuffle
from functools import partial

from fastai.metrics import accuracy_thresh
from fastai import fit, GradientClipping, Learner

import torch
from torch.optim import Adam
import torch.nn as nn
import pydash as _
import torch.nn.functional as F

from metrics import RankingMetricRecorder, recall, precision, f1
from losses import hinge_loss

def _get_loss_function(use_pairwise_loss):
  if use_pairwise_loss:
    return hinge_loss
  else:
    return F.mse_loss

def train_model(model, model_data, train_ranking_dataset, test_ranking_dataset, use_pairwise_loss):
  model = nn.DataParallel(model)
  loss = _get_loss_function(use_pairwise_loss)
  metrics = []
  num_epochs = 10000
  callbacks = [RankingMetricRecorder(model_data.device,
                                     model.module.pointwise_scorer if hasattr(model.module, 'pointwise_scorer') else model,
                                     train_ranking_dataset,
                                     test_ranking_dataset)]
  callback_fns=[partial(GradientClipping, clip=0.1)]
  print("Training:")
  learner = Learner(model_data,
                    model,
                    opt_fn=Adam,
                    loss_fn=loss,
                    metrics=metrics,
                    callbacks=callbacks,
                    callback_fns=callback_fns,
                    wd=0.0)
  learner.fit(num_epochs)
  torch.save(model.state_dict(), './model_save')
