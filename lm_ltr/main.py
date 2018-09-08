import pickle
import random

import pydash as _
import torch
import torch.nn as nn

from embedding_loaders import get_glove_lookup, init_embedding
from fetchers import get_raw_documents, get_supervised_raw_data, get_weak_raw_data, read_or_cache, read_cache
from pointwise_scorer import PointwiseScorer
from preprocessing import preprocess_raw_data, preprocess_texts
from train_model import train_model
from utils import with_negative_samples

def get_model(query_token_embed_len: int,
              document_token_embed_len: int,
              query_token_index_lookup: dict,
              document_token_index_lookup: dict) -> nn.Module:
  glove_lookup = get_glove_lookup()
  num_query_tokens = len(query_token_index_lookup)
  num_document_tokens = len(document_token_index_lookup)
  query_token_embeds = init_embedding(glove_lookup,
                                      query_token_index_lookup,
                                      num_query_tokens,
                                      query_token_embed_len)
  document_token_embeds = init_embedding(glove_lookup,
                                         document_token_index_lookup,
                                         num_document_tokens,
                                         document_token_embed_len)
  return PointwiseScorer(query_token_embeds, document_token_embeds)

def prepare_data():
  print('Loading mappings')
  with open('./document_ids.pkl', 'rb') as fh:
    document_title_id_mapping = pickle.load(fh)
    id_document_title_mapping = {document_title_id: document_title for document_title, document_title_id in _.to_pairs(document_title_id_mapping)}
  with open('./query_ids.pkl', 'rb') as fh:
    query_id_mapping = pickle.load(fh)
    id_query_mapping = {query_id: query for query, query_id in _.to_pairs(query_id_mapping)}
  print('Loading raw documents')
  raw_documents = read_or_cache('./raw_documents.pkl',
                                lambda: get_raw_documents(id_document_title_mapping))
  documents, document_token_lookup = preprocess_texts(raw_documents)
  size_train_queries = 0.8
  train_query_ids = random.sample(list(id_query_mapping.keys()),
                                  int(size_train_queries * len(id_query_mapping)))
  print('Loading weak data (from Indri output)')
  weak_raw_data = read_or_cache('./weak_raw_data.pkl',
                                lambda: get_weak_raw_data(id_query_mapping, train_query_ids))
  train_data, query_token_lookup = preprocess_raw_data(weak_raw_data)
  test_queries = [id_query_mapping[id] for id in set(id_query_mapping.keys()) - set(train_query_ids)]
  print('Loading supervised data (from mention-entity pairs)')
  supervised_raw_data = read_or_cache('./supervised_raw_data.pkl',
                                      lambda: get_supervised_raw_data(document_title_id_mapping, test_queries))
  test_data, __ = preprocess_raw_data(supervised_raw_data,
                                      query_token_lookup=query_token_lookup)
  return documents, train_data, test_data, query_token_lookup, document_token_lookup

def main():
  documents, train_data, test_data, query_token_lookup, document_token_lookup = read_cache('./prepared_data.pkl', prepare_data)
  documents = [doc[:100] for doc in documents]
  query_token_embed_len = 100
  document_token_embed_len = 100
  model = get_model(query_token_embed_len,
                    document_token_embed_len,
                    query_token_lookup,
                    document_token_lookup)
  train_model(model, documents, train_data, test_data)

if __name__ == "__main__":
  import ipdb
  import traceback
  import sys

  try:
    main()
  except: # pylint: disable=bare-except
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
  ipdb.post_mortem(tb)
