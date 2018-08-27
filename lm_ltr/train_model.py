import random

from fastai.model import fit
from fastai.dataset import ModelData
import pydash as _
from torch.optim import Adam
import torch.nn.functional as F

from data_wrappers import build_dataloader
from preprocessing import preprocess_texts

def get_negative_samples(num_query_tokens, num_negative_samples, max_len=4):
  result = []
  for i in range(num_negative_samples):
    query_len = random.randint(0, max_len)
    query = random.choices(range(2, num_query_tokens), k=query_len)
    result.append(query)
  return result

def train_model(model, raw_data) -> None:
  raw_documents, raw_train_queries, train_document_ids, raw_test_queries, test_document_ids = _.map_(['documents',
                                                                                                      'train_queries',
                                                                                                      'train_document_ids',
                                                                                                      'test_queries',
                                                                                                      'test_document_ids'],
                                                                                                     lambda key: raw_data[key])
  print('Preprocessing datasets')
  num_negative_train_samples = len(raw_train_queries) * 10
  num_negative_test_samples = len(raw_test_queries) * 10
  documents, document_token_lookup = preprocess_texts(raw_documents)
  processed_train_queries, query_token_lookup = preprocess_texts(raw_train_queries)
  processed_test_queries, __ = preprocess_texts(raw_test_queries, query_token_lookup)
  train_queries = processed_train_queries + get_negative_samples(len(query_token_lookup),
                                                                 num_negative_train_samples)
  test_queries = processed_test_queries + get_negative_samples(len(query_token_lookup),
                                                               num_negative_test_samples)
  train_labels = [1] * len(processed_train_queries) + [0] * num_negative_train_samples
  test_labels = [1] * len(processed_test_queries) + [0] * num_negative_test_samples
  print('Training')
  model_data = ModelData('./rows',
                         build_dataloader(documents, train_queries, train_labels),
                         build_dataloader(documents, test_queries, test_labels))
  fit(model,
      model_data,
      1,
      Adam(model.parameters()),
      F.cross_entropy)
