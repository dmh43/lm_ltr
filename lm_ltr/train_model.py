from random import shuffle

from fastai.dataset import ModelData
from fastai.metrics import accuracy_thresh
from fastai.model import fit

import torch
from torch.optim import Adam
import torch.nn as nn
import pydash as _
import torch.nn.functional as F

from data_wrappers import build_query_dataloader
from metrics import RankingMetricRecorder, recall, precision, f1
from validate_model import validate_model

def train_model(model, documents, train_data, test_data):
  print('Training')
  train_dl = build_query_dataloader(documents, train_data)
  test_dl = build_query_dataloader(documents, test_data)
  ranking_test_dl = build_ranking_dataloader(documents, test_data)
  model_data = ModelData('./rows', train_dl, test_dl)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = nn.DataParallel(model).to(device)
  metrics = []
  print("Untrained Model:")
  print(validate_model(model, F.mse_loss, test_dl, metrics))
  print("Training:")
  fit(model,
      model_data,
      100,
      Adam(list(filter(lambda p: p.requires_grad, model.parameters())),
           weight_decay=1.0),
      F.mse_loss)
  torch.save(model.state_dict(), './model_save')
