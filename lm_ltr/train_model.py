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
from preprocessing import pad_to_max_len

def train_model(model, documents, train_data, test_data):
  print('Training')
  train_dl = build_dataloader(documents, train_data)
  test_dl = build_dataloader(documents, test_data)
  model_data = ModelData('./rows',
                         train_dl,
                         test_dl)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = nn.DataParallel(model).to(device)
  metric_callback = RankingMetricRecorder(model,
                                          torch.tensor(pad_to_max_len(documents)),
                                          train_dl,
                                          test_dl,
                                          k=10)
  fit(model,
      model_data,
      100,
      Adam(list(filter(lambda p: p.requires_grad, model.parameters())),
           weight_decay=1.0),
      F.binary_cross_entropy_with_logits,
      metrics=[accuracy_thresh(0.5), recall, precision, f1],
      callbacks=[metric_callback])
