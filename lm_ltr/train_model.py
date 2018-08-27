from fastai.model import fit
from fastai.dataset import ModelData
import pydash as _
from torch.optim import Adam
import torch.nn.functional as F

from data_wrappers import build_dataloader
from preprocessing import preprocess_texts

def train_model(model, raw_data) -> None:
  raw_documents, raw_train_queries, train_labels, raw_test_queries, test_labels = _.map_(['documents',
                                                                                          'train_queries',
                                                                                          'train_labels',
                                                                                          'test_queries',
                                                                                          'test_labels'],
                                                                                         lambda key: raw_data[key])
  print('Preprocessing datasets')
  documents, document_term_lookup = preprocess_texts(raw_documents)
  train_queries, query_term_lookup = preprocess_texts(raw_train_queries)
  test_queries, __ = preprocess_texts(raw_test_queries, query_term_lookup)
  print('Training')
  model_data = ModelData('./rows',
                         build_dataloader(documents, train_queries, train_labels),
                         build_dataloader(documents, test_queries, test_labels))
  fit(model,
      model_data,
      1,
      Adam(model.parameters()),
      F.cross_entropy)
