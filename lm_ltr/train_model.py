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

def train_model(model,
                model_data,
                train_ranking_dataset,
                test_ranking_dataset,
                train_params,
                model_params,
                experiment):
  loss = F.mse_loss if train_params.use_pointwise_loss else hinge_loss
  model = nn.DataParallel(model)
  metrics = []
  callbacks = [RankingMetricRecorder(model_data.device,
                                     model.module.pointwise_scorer if hasattr(model.module, 'pointwise_scorer') else model,
                                     train_ranking_dataset,
                                     test_ranking_dataset,
                                     experiment,
                                     doc_chunk_size=train_params.batch_size if model_params.use_pretrained_doc_encoder else -1)]
  callback_fns = []
  if train_params.use_gradient_clipping:
    callback_fns.append(partial(GradientClipping, clip=train_params.gradient_clipping_norm))
  callback_fns.extend([partial(PlottingRecorder, experiment),
                       partial(LossesRecorder, experiment)])
  print("Training:")
  learner = Learner(model_data,
                    model,
                    opt_fn=Adam,
                    loss_fn=loss,
                    metrics=metrics,
                    callbacks=callbacks,
                    callback_fns=callback_fns,
                    wd=train_params.weight_decay)
  learner.fit(train_params.num_epochs, lr=train_params.learning_rate)
  torch.save(model.state_dict(), './model_save_' + experiment.model_name)
