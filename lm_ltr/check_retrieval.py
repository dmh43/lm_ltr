import pickle
import random
import time

import pydash as _
import torch
import torch.nn as nn

from .embedding_loaders import get_glove_lookup, init_embedding
from .fetchers import read_cache
from .preprocessing import pad_to_max_len
from .data_wrappers import get_top_k, score_documents_tfidf, _get_tfidf_transformer_and_matrix, score_documents_embed

def _raise_exception():
  raise FileNotFoundError('make sure that prepared_data.pkl exists')

def check_tfidf_method():
  documents, weak_data, sup_data, query_token_lookup, document_token_lookup = read_cache('./prepared_data.pkl', _raise_exception)
  documents = torch.tensor(pad_to_max_len(documents), dtype=torch.long)
  query_document_token_mapping = {idx: document_token_lookup.get(token) or document_token_lookup['<unk>'] for token, idx in query_token_lookup.items()}
  query_top_doc_pairs = [[row['query'], row['doc_id']] for row in weak_data if row['rank'] == 0]
  retrieved = 0
  xformer, tfidf_docs = _get_tfidf_transformer_and_matrix(documents)
  for query, doc_id in query_top_doc_pairs:
    scores = score_documents_tfidf(query_document_token_mapping, tfidf_docs, query)
    top_k = get_top_k(scores, k=1000)
    retrieved += (doc_id == top_k).any().item()
  print(retrieved/len(query_top_doc_pairs))

def check_embed_method():
  documents, weak_data, sup_data, query_token_lookup, document_token_lookup = read_cache('./prepared_data.pkl', _raise_exception)
  glove_lookup = get_glove_lookup()
  num_query_tokens = len(query_token_lookup)
  num_doc_tokens = len(document_token_lookup)
  query_token_embeds = init_embedding(glove_lookup,
                                      query_token_lookup,
                                      num_query_tokens,
                                      100)
  document_token_embeds = init_embedding(glove_lookup,
                                         document_token_lookup,
                                         num_doc_tokens,
                                         100)
  documents = torch.tensor(pad_to_max_len(documents), dtype=torch.long)
  query_top_doc_pairs = [[row['query'], row['doc_id']] for row in weak_data if row['rank'] == 0]
  queries, doc_ids = list(zip(*query_top_doc_pairs))
  queries = torch.tensor(pad_to_max_len(queries), dtype=torch.long)
  device = torch.device("cuda")
  scores = score_documents_embed(document_token_embeds, query_token_embeds, documents, queries, device)
  sorted_scores, idxs = torch.sort(scores, dim=1, descending=True)
  retrieved = 0
  for doc_id, top_k in zip(doc_ids, idxs[:, :1000]):
    retrieved += (doc_id == top_k).any().item()
  print(retrieved/len(query_top_doc_pairs))


def main():
  check_tfidf_method()

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
