from typing import List

import pydash as _
import torch
import torch.nn as nn

from embedding_loaders import get_glove_lookup, init_embedding
from eval_model import eval_model
from fetchers import get_rows, read_from_file, write_to_file
from lm_scorer import LMScorer
from preprocessing import preprocess_raw_data, get_raw_train_test
from train_model import train_model

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
  return LMScorer(query_token_embeds, document_token_embeds)

def main():
  print('Getting dataset')
  try:
    preprocessed_data = read_from_file('./processed')
  except:
    try:
      rows = get_rows()
      write_to_file('./rows', rows)
    except:
      rows = read_from_file('./rows')
    raw_data = get_raw_train_test(rows)
    print('Preprocessing datasets')
    preprocessed_data = preprocess_raw_data(raw_data)
    write_to_file('./processed', preprocessed_data)
  documents             = preprocessed_data['documents']
  document_token_lookup = preprocessed_data['document_token_lookup']
  query_token_lookup    = preprocessed_data['query_token_lookup']
  train_queries         = preprocessed_data['train_queries']
  train_document_ids    = preprocessed_data['train_document_ids']
  train_labels          = preprocessed_data['train_labels']
  test_queries          = preprocessed_data['test_queries']
  test_document_ids     = preprocessed_data['test_document_ids']
  test_labels           = preprocessed_data['test_labels']
  query_token_embed_len = 100
  document_token_embed_len = 100
  model = get_model(query_token_embed_len,
                    document_token_embed_len,
                    query_token_lookup,
                    document_token_lookup)
  train_model(model, documents, train_queries, train_document_ids, train_labels, test_document_ids, test_queries, test_labels)
  eval_model(model, raw_data)


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
