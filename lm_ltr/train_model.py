from random import shuffle

from fastai.dataset import ModelData
from fastai.metrics import accuracy_thresh
from fastai.model import fit

import torch
from torch.optim import Adam
import torch.nn as nn
import pydash as _
import torch.nn.functional as F

from data_wrappers import build_query_dataloader, RankingDataset
from metrics import RankingMetricRecorder, recall, precision, f1
from validate_model import validate_model

def train_model(model, documents, train_dl, test_dl):
  print('Creating ranking datasets')
  train_ranking_dataset = RankingDataset(documents, train_dl.dataset.data)
  test_ranking_dataset = RankingDataset(documents, test_dl.dataset.data)
  model_data = ModelData('./rows', train_dl, test_dl)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = nn.DataParallel(model).to(device)
  metrics = []
  print("Untrained Model:")
  print(validate_model(model, F.mse_loss, test_dl, metrics))
  print("Training:")
  fit(model,
      model_data,
      10000,
      Adam(list(filter(lambda p: p.requires_grad, model.parameters())),
           weight_decay=0.0),
      F.mse_loss,
      callbacks=[RankingMetricRecorder(model, train_ranking_dataset, test_ranking_dataset)])
  torch.save(model.state_dict(), './model_save')
