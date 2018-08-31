from fastai.dataset import ModelData
from fastai.metrics import accuracy
from fastai.model import fit

import torch
from torch.optim import Adam
import torch.nn as nn
import pydash as _
import torch.nn.functional as F

from data_wrappers import build_dataloader

def train_model(model, documents, train_queries, train_document_ids, train_labels, test_document_ids, test_queries, test_labels) -> None:
  print('Training')
  model_data = ModelData('./rows',
                         build_dataloader(documents, train_queries, train_document_ids, train_labels),
                         build_dataloader(documents, test_queries, test_document_ids, test_labels))
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = nn.DataParallel(model).to(device)
  fit(model,
      model_data,
      100,
      Adam(list(filter(lambda p: p.requires_grad, model.parameters())),
           weight_decay=1.0),
      F.cross_entropy,
      metrics=[accuracy])
