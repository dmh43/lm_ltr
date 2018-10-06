from random import shuffle

from fastai.metrics import accuracy_thresh
from fastai.model import fit

import torch
from torch.optim import Adam
import torch.nn as nn
import pydash as _
import torch.nn.functional as F

from metrics import RankingMetricRecorder, recall, precision, f1
from validate_model import validate_model
from losses import hinge_loss

def _get_loss_function(use_pairwise_loss):
  if use_pairwise_loss:
    return hinge_loss
  else:
    return F.mse_loss

def train_model(model, model_data, train_ranking_dataset, test_ranking_dataset, use_pairwise_loss):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = nn.DataParallel(model).to(device)
  loss = _get_loss_function(use_pairwise_loss)
  metrics = []
  print("Training:")
  fit(model,
      model_data,
      10000,
      Adam(list(filter(lambda p: p.requires_grad, model.parameters())),
           weight_decay=0.01),
      loss,
      callbacks=[RankingMetricRecorder(device,
                                       model.module.pointwise_scorer if hasattr(model.module, 'pointwise_scorer') else model,
                                       train_ranking_dataset,
                                       test_ranking_dataset)])
  torch.save(model.state_dict(), './model_save')
