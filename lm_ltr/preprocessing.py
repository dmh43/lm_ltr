import ast
import random

import pydash as _
import numpy as np

import torch
import torch.nn.functional as F
from fastai.text import Tokenizer
from torch.nn.utils.rnn import pad_sequence

from .utils import append_at

pad_token_idx = 1
unk_token_idx = 0

def tokens_to_indexes(tokens, lookup=None, num_tokens=None, token_set=None):
  is_test = lookup is not None
  if lookup is None:
    lookup: dict = {'<unk>': unk_token_idx, '<pad>': pad_token_idx}
  result = []
  for tokens_chunk in tokens:
    tokens_to_parse = tokens_chunk if num_tokens is None else tokens_chunk[:num_tokens]
    chunk_result = []
    for token in tokens_to_parse:
      if (token_set is None) or (token in token_set):
        if is_test:
          chunk_result.append(lookup.get(token) or unk_token_idx)
        else:
          lookup[token] = lookup.get(token) or len(lookup)
          chunk_result.append(lookup[token])
      else:
        chunk_result.append(unk_token_idx)
    result.append(chunk_result)
  return result, lookup

def preprocess_texts(texts, token_lookup=None, num_tokens=None, token_set=None):
  tokenizer = Tokenizer()
  tokenized = tokenizer.process_all(texts)
  idx_texts, token_lookup = tokens_to_indexes(tokenized,
                                              token_lookup,
                                              num_tokens=num_tokens,
                                              token_set=token_set)
  return idx_texts, token_lookup

def pad_to_len(coll, max_len, pad_with=None):
  pad_with = pad_with if pad_with is not None else pad_token_idx
  return coll + [pad_with] * (max_len - len(coll)) if len(coll) < max_len else coll

def pad_to_max_len(elems, pad_with=None):
  pad_with = pad_with if pad_with is not None else pad_token_idx
  max_len = max(map(len, elems))
  return [elem + [pad_with] * (max_len - len(elem)) if len(elem) < max_len else elem for elem in elems]

def pad(batch, device=torch.device('cpu')):
  batch_lengths = torch.tensor(_.map_(batch, len),
                               dtype=torch.long,
                               device=device)
  return (pad_sequence(batch, batch_first=True, padding_value=1).to(device),
          batch_lengths)

def collate_query_samples(samples):
  x, rel = list(zip(*samples))
  x = list(zip(*x))
  query = pad_to_max_len(x[0])
  doc, lens = pad(x[1])
  return ((torch.tensor(query), doc, lens),
          torch.tensor(rel, dtype=torch.float32))

def collate_query_pairwise_samples(samples):
  x, rel = list(zip(*samples))
  x = list(zip(*x))
  query = pad_to_max_len(x[0])
  doc_1, lens_1 = pad(x[1])
  doc_2, lens_2 = pad(x[2])
  return ((torch.tensor(query), doc_1, doc_2, lens_1, lens_2),
          torch.tensor(rel, dtype=torch.float32))

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
  return torch.sigmoid(raw_info['score'])

def all_ones(raw_info):
  return 1.0

def sort_by_first(pairs):
  return sorted(pairs, key=lambda val: val[0])

def to_query_rankings_pairs(data):
  query_to_ranking = {}
  q_str_to_query = {}
  for row in data:
    q_str_to_query[str(row['query'])[1:-1]] = row['query']
    append_at(query_to_ranking, str(row['query'])[1:-1], row['doc_id'])
  querystr_ranking_pairs = _.to_pairs(query_to_ranking)
  return _.map_(querystr_ranking_pairs,
                lambda q_str_ranking: [q_str_to_query[q_str_ranking[0]],
                                       q_str_ranking[1]])

def create_id_lookup(names_or_titles):
  return dict(zip(names_or_titles,
                  range(len(names_or_titles))))

def prepare(lookup, title_to_id, token_lookup=None, num_tokens=None, token_set=None):
  id_to_title_lookup = _.invert(title_to_id)
  ids = range(len(id_to_title_lookup))
  contents = [lookup[id_to_title_lookup[id]] for id in ids]
  numericalized, token_lookup = preprocess_texts(contents,
                                                 token_lookup=token_lookup,
                                                 num_tokens=num_tokens,
                                                 token_set=token_set)
  return numericalized, token_lookup

def normalize_scores_query_wise(data):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  query_doc_info = {}
  for row in data:
    score = row.get('score') or 0.0
    append_at(query_doc_info, str(row['query'])[1:-1], [row['doc_id'], score, row['query']])
  normalized_data = []
  for doc_infos in query_doc_info.values():
    scores = torch.tensor([doc_score for doc_id, doc_score, query in doc_infos], device=device)
    query_score_total = torch.logsumexp(scores, 0)
    normalized_scores = scores - query_score_total
    normalized_data.extend([{'query': doc_info[2],
                             'doc_id': doc_info[0],
                             'score': score}
                            for doc_info, score in zip(doc_infos, normalized_scores.tolist())])
  return normalized_data

def process_rels(query_name_document_title_rels, document_title_to_id, query_name_to_id, queries):
  data = []
  for query_name, doc_titles in query_name_document_title_rels.items():
    if query_name not in query_name_to_id: continue
    query_id = query_name_to_id[query_name]
    query = queries[query_id]
    if query is None: continue
    data.extend([{'query': query,
                  'score': 1.0,
                  'doc_id': document_title_to_id[title]} for title in doc_titles if title in document_title_to_id])
  return data
