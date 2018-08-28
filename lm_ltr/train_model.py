from fastai.dataset import ModelData
from fastai.metrics import accuracy
from fastai.model import fit
from torch.optim import Adam
import pydash as _
import torch.nn.functional as F

from data_wrappers import build_dataloader

def train_model(model, documents, train_queries, train_labels, test_queries, test_labels) -> None:
  print('Training')
  model_data = ModelData('./rows',
                         build_dataloader(documents, train_queries, train_labels),
                         build_dataloader(documents, test_queries, test_labels))
  fit(model,
      model_data,
      1,
      Adam(model.parameters()),
      F.cross_entropy,
      metrics=[accuracy])
