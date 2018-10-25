import ast
import random

import pydash as _
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence
from fastai.text import Tokenizer

from .utils import append_at

pad_token_idx = 1
unk_token_idx = 0

def tokens_to_indexes(tokens, lookup=None, num_tokens=None):
  is_test = lookup is not None
  if lookup is None:
    lookup: dict = {'<unk>': unk_token_idx, '<pad>': pad_token_idx}
  result = []
  for tokens_chunk in tokens:
    tokens_to_parse = tokens_chunk if num_tokens is None else tokens_chunk[:num_tokens]
    chunk_result = []
    for token in tokens_to_parse:
      if is_test:
        chunk_result.append(lookup.get(token) or unk_token_idx)
      else:
        lookup[token] = lookup.get(token) or len(lookup)
        chunk_result.append(lookup[token])
    result.append(chunk_result)
  return result, lookup

def preprocess_texts(texts, token_lookup=None, num_tokens=None):
  tokenizer = Tokenizer()
  tokenized = tokenizer.process_all(texts)
  idx_texts, token_lookup = tokens_to_indexes(tokenized, token_lookup, num_tokens=num_tokens)
  return idx_texts, token_lookup

def pad_to_max_len(elems, pad_with=None):
  pad_with = pad_with if pad_with is not None else pad_token_idx
  max_len = max(map(len, elems))
  return [elem + [pad_with] * (max_len - len(elem)) if len(elem) < max_len else elem for elem in elems]

def collate_query_samples(samples):
  x, rel = list(zip(*samples))
  x = list(zip(*x))
  query = pad_to_max_len(x[0])
  documents = pad_to_max_len(x[1])
  return (torch.tensor(query), torch.tensor(documents)), torch.tensor(rel)

def pack(batch, device=torch.device('cpu')):
  batch_lengths = torch.tensor(_.map_(batch, len),
                               dtype=torch.long,
                               device=device)
  sorted_batch_lengths, batch_order = torch.sort(batch_lengths, descending=True)
  batch_range, unsort_batch_order = torch.sort(batch_order)
  sorted_batch = torch.tensor(batch,
                              dtype=torch.long,
                              device=device)[batch_order]
  return (pack_sequence(sorted_batch), unsort_batch_order)

def collate_query_pairwise_samples(samples):
  x, rel = list(zip(*samples))
  x = list(zip(*x))
  query = pad_to_max_len(x[0])
  packed_doc_1_and_order = pack(x[1])
  packed_doc_2_and_order = pack(x[2])
  return ((torch.tensor(query),
           packed_doc_1_and_order,
           packed_doc_2_and_order),
          torch.tensor(rel))

def get_negative_samples(num_query_tokens, num_negative_samples, max_len=4):
  result = []
  for i in range(num_negative_samples):
    query_len = random.randint(1, max_len)
    query = random.choices(range(2, num_query_tokens), k=query_len)
    result.append(query)
  return result

def inv_log_rank(raw_info):
  return 1.0 / np.log(raw_info['rank'] + 1)

def inv_rank(raw_info):
  return 1.0 / (raw_info['rank'] + 1)

def score(raw_info):
  return raw_info['score']

def exp_score(raw_info):
  return 2 ** raw_info['score']

def sigmoid_score(raw_info):
  return F.sigmoid(raw_info['score'])

def all_ones(raw_info):
  return 1.0

def preprocess_raw_data(raw_data, query_token_lookup=None):
  queries = [sample['query'] for sample in raw_data]
  tokens, lookup = preprocess_texts(queries, query_token_lookup)
  preprocessed_data = [_.assign({},
                                sample,
                                {'query': query_tokens}) for query_tokens, sample in zip(tokens, raw_data)]
  return preprocessed_data, lookup

def sort_by_first(pairs):
  return sorted(pairs, key=lambda val: val[0])

def to_query_rankings_pairs(data):
  query_to_ranking = {}
  for row in data:
    append_at(query_to_ranking, str(row['query'])[1:-1], row['doc_id'])
  querystr_ranking_pairs = _.to_pairs(query_to_ranking)
  return [[ast.literal_eval('[' + pair[0] + ']'), pair[1]] for pair in querystr_ranking_pairs]

def create_id_lookup(names_or_titles):
  return dict(zip(names_or_titles,
                  range(len(names_or_titles))))

def prepare(lookup, title_to_id, token_lookup=None, num_tokens=None):
  id_to_title_lookup = _.invert(title_to_id)
  ids = range(len(id_to_title_lookup))
  contents = [lookup[id_to_title_lookup[id]] for id in ids]
  numericalized, token_lookup = preprocess_texts(contents, token_lookup=token_lookup, num_tokens=num_tokens)
  return numericalized, token_lookup
