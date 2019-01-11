from collections import defaultdict, Counter
import ast
import random
import re

import pydash as _
import numpy as np

import torch
import torch.nn.functional as F
from fastai.text import Tokenizer, fix_html, spec_add_spaces, rm_useless_spaces
from torch.nn.utils.rnn import pad_sequence

from .utils import append_at

pad_token_idx = 1
unk_token_idx = 0

def tokens_to_indexes(tokens, lookup=None, num_tokens=None, token_set=None, drop_if_any_unk=False):
  def _append(coll, val, idx):
    if isinstance(coll, dict):
      coll[idx] = val
    else:
      coll.append(val)
  is_test = lookup is not None
  if lookup is None:
    lookup: dict = {'<unk>': unk_token_idx, '<pad>': pad_token_idx}
  result = [] if not drop_if_any_unk else {}
  for idx, tokens_chunk in enumerate(tokens):
    tokens_to_parse = tokens_chunk if num_tokens is None else tokens_chunk[:num_tokens]
    chunk_result = []
    for token in tokens_to_parse:
      if (token_set is None) or (token in token_set):
        if is_test:
          if drop_if_any_unk and lookup.get(token) is None:
            chunk_result = []
            break
          chunk_result.append(lookup.get(token) or unk_token_idx)
        else:
          lookup[token] = lookup.get(token) or len(lookup)
          chunk_result.append(lookup[token])
      else:
        if drop_if_any_unk:
          chunk_result = []
          break
        chunk_result.append(unk_token_idx)
    if len(chunk_result) > 0:
      _append(result, chunk_result, idx)
  return result, lookup

def handle_caps(t:str) -> str:
  "Replace words in all caps in `t`."
  res = []
  for s in re.findall(r'\w+|\W+', t):
    res += ([f' ', s.lower()] if (s.isupper() and (len(s)>2)) else [s.lower()])
  return ''.join(res)

def preprocess_texts(texts, token_lookup=None, num_tokens=None, token_set=None, drop_if_any_unk=False):
  tokenizer = Tokenizer(rules=[handle_caps, fix_html, spec_add_spaces, rm_useless_spaces])
  tokenized = tokenizer.process_all(texts)
  idx_texts, token_lookup = tokens_to_indexes(tokenized,
                                              token_lookup,
                                              num_tokens=num_tokens,
                                              token_set=token_set,
                                              drop_if_any_unk=drop_if_any_unk)
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

def collate_query_samples(samples, use_bow_model=False):
  x, rel = list(zip(*samples))
  x = list(zip(*x))
  query = pad_to_max_len(x[0])
  doc, lens = list(zip(*x[1]))
  doc_score = x[2]
  return ((torch.tensor(query),
           torch.stack(doc) if not use_bow_model else _collate_bow_doc(doc),
           torch.stack(lens),
           torch.tensor(doc_score)),
          torch.tensor(rel, dtype=torch.float32))

def _collate_bow_doc(bow_doc):
  terms = []
  cnts = []
  max_len = 0
  for doc in bow_doc:
    doc_terms = list(doc.keys())
    max_len = max(max_len, len(doc_terms))
    terms.append(torch.tensor(doc_terms))
    cnts.append(torch.tensor([doc[term] for term in doc_terms]))
  terms = torch.stack(pad_to_len(terms, max_len, pad_with=1))
  cnts = torch.stack(pad_to_len(cnts, max_len))
  return terms, cnts

def collate_query_pairwise_samples(samples, use_bow_model=False):
  x, rel = list(zip(*samples))
  x = list(zip(*x))
  query = pad_to_max_len(x[0])
  doc_1, lens_1 = list(zip(*x[1]))
  doc_2, lens_2 = list(zip(*x[2]))
  doc_1_score = x[3]
  doc_2_score = x[4]
  return ((torch.tensor(query),
           torch.stack(doc_1) if not use_bow_model else _collate_bow_doc(doc_1),
           torch.stack(doc_2) if not use_bow_model else _collate_bow_doc(doc_2),
           torch.stack(lens_1),
           torch.stack(lens_2),
           torch.tensor(doc_1_score),
           torch.tensor(doc_2_score)),
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

def to_query_rankings_pairs(data, limit=None):
  query_to_ranking = {}
  q_str_to_query = {}
  for row in data:
    q_str_to_query[str(row['query'])[1:-1]] = row['query']
    if (limit is None) or (len(query_to_ranking.get(str(row['query'])[1:-1]) or []) <= limit):
      append_at(query_to_ranking, str(row['query'])[1:-1], row['doc_id'])
  querystr_ranking_pairs = _.to_pairs(query_to_ranking)
  return _.map_(querystr_ranking_pairs,
                lambda q_str_ranking: [q_str_to_query[q_str_ranking[0]],
                                       q_str_ranking[1]])

def create_id_lookup(names_or_titles):
  return dict(zip(names_or_titles,
                  range(len(names_or_titles))))

def prepare(lookup, title_to_id, token_lookup=None, num_tokens=None, token_set=None, drop_if_any_unk=False):
  id_to_title_lookup = _.invert(title_to_id)
  ids = range(len(id_to_title_lookup))
  contents = [lookup[id_to_title_lookup[id]] for id in ids]
  numericalized, token_lookup = preprocess_texts(contents,
                                                 token_lookup=token_lookup,
                                                 num_tokens=num_tokens,
                                                 token_set=token_set,
                                                 drop_if_any_unk=drop_if_any_unk)
  return numericalized, token_lookup

def prepare_fs(lookup,
               title_to_id,
               token_lookup=None,
               num_tokens=None,
               token_set=None,
               drop_if_any_unk=False):
  id_to_title_lookup = _.invert(title_to_id)
  ids = range(len(id_to_title_lookup))
  contents = [lookup[id_to_title_lookup[id]] for id in ids]
  numericalized, token_lookup = preprocess_texts(contents,
                                                 token_lookup=token_lookup,
                                                 token_set=token_set,
                                                 drop_if_any_unk=drop_if_any_unk)
  numericalized_fs = [Counter(doc) for doc in numericalized]
  return numericalized_fs, token_lookup

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

def get_normalized_score_lookup(data):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  query_doc_info = {}
  for row in data:
    score = row.get('score') or 0.0
    append_at(query_doc_info, str(row['query'])[1:-1], [row['doc_id'], score, row['query']])
  normalized_lookup = defaultdict(lambda: {})
  for doc_infos in query_doc_info.values():
    scores = torch.tensor([doc_score for doc_id, doc_score, query in doc_infos], device=device)
    query_score_total = torch.logsumexp(scores, 0)
    normalized_scores = scores - query_score_total
    for doc_info, score in zip(doc_infos, normalized_scores.tolist()):
      normalized_lookup[tuple(doc_info[2])][doc_info[0]] = score
  return dict(normalized_lookup)

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
