from typing import List
import random

import torch
from fastai.text import Tokenizer

pad_token_idx = 1
unk_token_idx = 0

def tokens_to_indexes(tokens: List[List[str]], lookup=None):
  is_test = lookup is not None
  if lookup is None:
    lookup: dict = {'<unk>': unk_token_idx, '<pad>': pad_token_idx}
  result = []
  for tokens_chunk in tokens:
    chunk_result = []
    for token in tokens_chunk:
      if is_test:
        chunk_result.append(lookup.get(token) or unk_token_idx)
      else:
        lookup[token] = lookup.get(token) or len(lookup)
        chunk_result.append(lookup[token])
    result.append(chunk_result)
  return result, lookup

def get_raw_train_test(rows, size=0.8):
  documents = []
  query_document_id_lookup = {}
  for row in rows:
    if query_document_id_lookup.get(row['query']) is None:
      query_document_id_lookup[row['query']] = len(documents)
      documents.append(row['document'])
  samples = _.to_pairs(query_document_id_lookup)
  num_train_samples = int(size * len(samples))
  random.shuffle(samples)
  queries, document_ids = list(zip(*samples))
  return {'train_queries': queries[:num_train_samples],
          'test_queries': queries[num_train_samples:],
          'train_document_ids': document_ids[:num_train_samples],
          'test_document_ids': document_ids[num_train_samples:],
          'documents': documents}

def preprocess_texts(texts: List[str], token_lookup=None):
  tokenizer = Tokenizer()
  tokenized = [tokenizer.proc_text(q) for q in texts]
  idx_texts, token_lookup = tokens_to_indexes(tokenized, token_lookup)
  return idx_texts, token_lookup

def pad_to_max_len(elems: List[List[int]]):
  max_len = max(map(len, elems))
  return [elem + [pad_token_idx] * (max_len - len(elem)) if len(elem) < max_len else elem for elem in elems]

def collate(samples):
  x, y = list(zip(*samples))
  x = list(zip(*x))
  return torch.tensor(pad_to_max_len(x[0])), x[1], torch.tensor(y)

def get_negative_samples(num_query_tokens, num_negative_samples, max_len=4):
  result = []
  for i in range(num_negative_samples):
    query_len = random.randint(1, max_len)
    query = random.choices(range(2, num_query_tokens), k=query_len)
    result.append(query)
  return result

def preprocess_raw_data(raw_data):
  raw_documents, raw_train_queries, train_document_ids, raw_test_queries, test_document_ids = _.map_(['documents',
                                                                                                      'train_queries',
                                                                                                      'train_document_ids',
                                                                                                      'test_queries',
                                                                                                      'test_document_ids'],
                                                                                                     lambda key: raw_data[key])
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
  return {'documents':             documents,
          'document_token_lookup': document_token_lookup,
          'query_token_lookup':    query_token_lookup,
          'train_queries':         train_queries,
          'train_labels':          train_labels,
          'test_queries':          test_queries,
          'test_labels':           test_labels}
