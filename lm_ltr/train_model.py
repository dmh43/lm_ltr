from random import shuffle

from fastai.dataset import ModelData
from fastai.metrics import accuracy_thresh
from fastai.model import fit

import torch
from torch.optim import Adam
import torch.nn as nn
import pydash as _
import torch.nn.functional as F

from data_wrappers import build_dataloader
from metrics import RankingMetricRecorder, recall, precision, f1

def train_model(model, documents, train_data, test_data):
  print('Training')
  train_dl = build_dataloader(documents, train_data)
  test_dl = build_dataloader(documents, test_data)
  model_data = ModelData('./rows', train_dl, test_dl)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = nn.DataParallel(model).to(device)
  print("Untrained Model:")
  fit(model,
      ModelData('./rows', build_dataloader(documents, train_data[:1]), test_dl),
      1,
      Adam(list(filter(lambda p: p.requires_grad, model.parameters())),
           weight_decay=1.0),
      F.mse_loss)
  print("Training:")
  fit(model,
      model_data,
      100,
      Adam(list(filter(lambda p: p.requires_grad, model.parameters())),
           weight_decay=1.0),
      F.mse_loss)
  torch.save(model.state_dict(), './model_save')
