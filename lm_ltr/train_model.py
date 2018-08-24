from fastai.model import fit
import pydash as _
from torch.optim import Adam
import torch.nn.functional as F

from data_wrappers import build_dataloader

def train_model(model, data) -> None:
  documents, train_queries, train_labels, test_queries, test_labels = _.map_(['document',
                                                                              'train_queries',
                                                                              'train_labels',
                                                                              'test_queries',
                                                                              'test_labels'],
                                                                             lambda key: data[key])
  fit(model,
      build_dataloader(documents, train_queries, train_labels),
      build_dataloader(documents, test_queries, test_labels),
      Adam,
      F.cross_entropy)
