import pickle
import random

import pydash as _
import torch
import torch.nn as nn

from embedding_loaders import get_glove_lookup, init_embedding
from fetchers import get_raw_documents, get_supervised_raw_data, get_weak_raw_data, read_or_cache, read_cache
from pointwise_scorer import PointwiseScorer
from preprocessing import preprocess_raw_data, preprocess_texts, all_ones, score
from data_wrappers import build_term_matching_dataloader, build_query_dataloader
from train_model import train_model
from utils import with_negative_samples
from term_matching_scorer import TermMatchingScorer

def get_model_and_dls(query_token_embed_len,
                      document_token_embed_len,
                      query_token_index_lookup,
                      document_token_index_lookup,
                      documents,
                      weak_data,
                      use_term_matching=False):
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
  train_data = weak_data[:int(len(weak_data) * 0.8)]
  test_data = weak_data[int(len(weak_data) * 0.8):]
  if use_term_matching:
    query_document_token_mapping = {idx: document_token_index_lookup.get(token) or document_token_index_lookup['<unk>'] for token, idx in query_token_index_lookup.items()}
    train_dl = build_term_matching_dataloader(query_document_token_mapping, documents, train_data)
    test_dl = build_term_matching_dataloader(query_document_token_mapping, documents, test_data)
    scorer = TermMatchingScorer(query_document_token_mapping)
  else:
    train_dl = build_query_dataloader(documents, train_data)
    test_dl = build_query_dataloader(documents, test_data)
    scorer = PointwiseScorer(query_token_embeds, document_token_embeds)
  return scorer, train_dl, test_dl

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
  train_data, query_token_lookup = preprocess_raw_data(weak_raw_data, rel_method=score)
  test_queries = [id_query_mapping[id] for id in set(id_query_mapping.keys()) - set(train_query_ids)]
  print('Loading supervised data (from mention-entity pairs)')
  supervised_raw_data = read_or_cache('./supervised_raw_data.pkl',
                                      lambda: get_supervised_raw_data(document_title_id_mapping, test_queries))
  test_data, __ = preprocess_raw_data(supervised_raw_data,
                                      query_token_lookup=query_token_lookup,
                                      rel_method=all_ones)
  return documents, train_data, test_data, query_token_lookup, document_token_lookup

def main():
  documents, weak_data, sup_data, query_token_lookup, document_token_lookup = read_cache('./prepared_data.pkl', prepare_data)
  documents = [doc[:100] for doc in documents]
  query_token_embed_len = 100
  document_token_embed_len = 100
  use_term_matching = True
  model, train_dl, test_dl = get_model_and_dls(query_token_embed_len,
                                               document_token_embed_len,
                                               query_token_lookup,
                                               document_token_lookup,
                                               documents,
                                               weak_data,
                                               use_term_matching=use_term_matching)
  train_model(model, documents, train_dl, test_dl)

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
