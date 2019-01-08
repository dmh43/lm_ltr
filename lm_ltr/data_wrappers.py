import ast
from random import sample, randint, choice
from functools import reduce
import pydash as _

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, Sampler, SequentialSampler

import scipy.sparse as sp
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer

from .preprocessing import collate_query_samples, collate_query_pairwise_samples, to_query_rankings_pairs, pad_to_max_len, all_ones, score, inv_log_rank, inv_rank, exp_score, pad
from .utils import append_at, to_list
from .fetchers import read_cache

def remap_if_exists(tokens, lookup):
  if lookup:
    return [lookup[token] for token in to_list(tokens)]
  else:
    return tokens

class QueryDataset(Dataset):
  def __init__(self,
               documents,
               data,
               rel_method=score,
               num_doc_tokens=100,
               rankings=None,
               query_tok_to_doc_tok=None,
               use_doc_out=False,
               normalized_score_lookup=None):
    self.documents = documents
    self.padded_docs = pad([torch.tensor(doc[:num_doc_tokens]) for doc in documents])
    self.data = data
    self.rel_method = rel_method
    self.rankings = rankings if rankings is not None else to_query_rankings_pairs(data)
    self.num_doc_tokens = num_doc_tokens
    self.query_tok_to_doc_tok = query_tok_to_doc_tok
    self.use_doc_out = use_doc_out
    self.normalized_score_lookup = normalized_score_lookup

  def _get_document(self, elem_idx):
    if self.use_doc_out:
      return torch.tensor(elem_idx), torch.tensor(0)
    else:
      return self.padded_docs[0][elem_idx], self.padded_docs[1][elem_idx]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    query = remap_if_exists(self.data[idx]['query'], self.query_tok_to_doc_tok)
    doc_id = self.data[idx]['doc_id']
    if self.normalized_score_lookup is not None:
      doc_score = self.normalized_score_lookup[tuple(query)][doc_id]
    else:
      doc_score = 0.0
    return ((query, self._get_document(doc_id), doc_score),
            self.rel_method(self.data[idx]))

def _shuffle_doc_doc_ids(documents, doc_ids):
  shuffled_doc_ids_for_batch = torch.randperm(len(doc_ids))
  shuffled_documents = [documents[i] for i in shuffled_doc_ids_for_batch]
  return shuffled_documents, doc_ids[shuffled_doc_ids_for_batch]

class RankingDataset(Dataset):
  def __init__(self,
               documents,
               rankings,
               relevant=None,
               num_doc_tokens=100,
               k=10,
               query_tok_to_doc_tok=None,
               use_doc_out=False,
               num_to_rank=1000,
               cheat=False,
               normalized_score_lookup=None):
    self.rankings = rankings
    self.documents = documents
    self.short_docs = [torch.tensor(doc[:num_doc_tokens]) for doc in documents]
    self.k = k
    self.num_doc_tokens = num_doc_tokens
    self.num_to_rank = num_to_rank
    self.is_test = relevant is not None
    self.relevant = relevant
    self.query_tok_to_doc_tok = query_tok_to_doc_tok
    self.use_doc_out = use_doc_out
    self.cheat = cheat
    self.normalized_score_lookup = normalized_score_lookup
    if self.is_test:
      self.rel_by_q_str = {str(query)[1:-1]: [query, rel] for query, rel in self.relevant}
      self.q_strs = list(set(rankings.keys()).intersection(set(self.rel_by_q_str.keys())))

  def __len__(self):
    return len(self.rankings)

  def _get_train_item(self, idx):
    query, ranking = self.rankings[idx]
    query = remap_if_exists(query, self.query_tok_to_doc_tok)
    relevant = set(ranking[:self.k])
    if len(ranking) < self.num_to_rank:
      neg_samples = sample(set(range(len(self.documents))) - set(ranking),
                           self.num_to_rank - len(ranking))
      ranking_with_neg = ranking + neg_samples
    else:
      ranking_with_neg = ranking[:self.num_to_rank]
    documents, doc_ids = _shuffle_doc_doc_ids([self.short_docs[idx] for idx in ranking_with_neg],
                                              torch.tensor(ranking_with_neg, dtype=torch.long))
    if self.normalized_score_lookup is not None:
      smallest_score = min(self.normalized_score_lookup[tuple(query)].values())
      doc_scores = torch.tensor([self.normalized_score_lookup[tuple(query)][doc_id]
                                 if doc_id in self.normalized_score_lookup[tuple(query)] else smallest_score
                                 for doc_id in doc_ids.tolist()])
    else:
      doc_scores = torch.zeros(len(documents), dtype=torch.float32)
    return {'query': torch.tensor(query, dtype=torch.long),
            'documents': documents if not self.use_doc_out else doc_ids,
            'doc_ids': doc_ids,
            'ranking': ranking[:self.k],
            'relevant': relevant,
            'doc_scores': doc_scores}

  def _get_test_item(self, idx):
    q_str = self.q_strs[idx]
    query, relevant = self.rel_by_q_str[q_str]
    q_str = str(query)[1:-1]
    query = remap_if_exists(query, self.query_tok_to_doc_tok)
    relevant = set(relevant)
    ranking = self.rankings[q_str][:self.num_to_rank]
    if self.cheat:
      lookup_ranking = set(ranking)
      for doc_id in relevant:
        if doc_id not in lookup_ranking:
          ranking.append(doc_id)
    documents, doc_ids = _shuffle_doc_doc_ids([self.short_docs[doc_id] for doc_id in ranking],
                                              torch.tensor(ranking, dtype=torch.long))
    if self.normalized_score_lookup is not None:
      smallest_score = min(self.normalized_score_lookup[tuple(query)].values())
      doc_scores = torch.tensor([self.normalized_score_lookup[tuple(query)][doc_id]
                                 if doc_id in self.normalized_score_lookup[tuple(query)] else smallest_score
                                 for doc_id in doc_ids.tolist()])
    else:
      doc_scores = torch.zeros(len(documents), dtype=torch.float32)
    return {'query': torch.tensor(query, dtype=torch.long),
            'documents': documents if not self.use_doc_out else doc_ids,
            'doc_ids': doc_ids,
            'ranking': self.rel_by_q_str[q_str][1][:self.k],
            'relevant': relevant,
            'doc_scores': doc_scores}

  def __getitem__(self, idx):
    return self._get_test_item(idx) if self.is_test else self._get_train_item(idx)

def _get_nth_pair(rankings, cumu_num_pairs, idx, use_variable_loss=False):
  ranking_idx = np.searchsorted(cumu_num_pairs, idx, side='right')
  offset = idx - cumu_num_pairs[ranking_idx - 1] if ranking_idx != 0 else idx
  query = rankings[ranking_idx][0]
  doc_ids = rankings[ranking_idx][1]
  num_doc_ids = len(doc_ids)
  doc_1_idx = offset // (num_doc_ids - 1)
  doc_2_idx = offset % (num_doc_ids - 1)
  if doc_2_idx >= doc_1_idx:
    doc_2_idx += 1
  if use_variable_loss:
    return {'query': query,
            'doc_id_1': doc_ids[doc_1_idx],
            'doc_id_2': doc_ids[doc_2_idx],
            'order_int': (doc_2_idx - doc_1_idx) / (len(doc_ids) ** 2 - len(doc_ids))}
  else:
    return {'query': query,
            'doc_id_1': doc_ids[doc_1_idx],
            'doc_id_2': doc_ids[doc_2_idx],
            'order_int': 1 if doc_1_idx < doc_2_idx else -1}

def _get_nth_pair_bin_rankings(rankings, cumu_num_pairs, bin_rankings, idx, use_variable_loss=False):
  ranking_idx = np.searchsorted(cumu_num_pairs, idx, side='right')
  offset = idx - cumu_num_pairs[ranking_idx - 1] if ranking_idx != 0 else idx
  query = rankings[ranking_idx][0]
  doc_ids = rankings[ranking_idx][1]
  doc_1_idx = 0
  doc_2_idx = offset + 1
  if use_variable_loss:
    return {'query': query,
            'doc_id_1': doc_ids[doc_1_idx],
            'doc_id_2': doc_ids[doc_2_idx],
            'order_int': (doc_2_idx - doc_1_idx) / (len(doc_ids) ** 2 - len(doc_ids))}
  else:
    return {'query': query,
            'doc_id_1': doc_ids[doc_1_idx],
            'doc_id_2': doc_ids[doc_2_idx],
            'order_int': 1 if doc_1_idx < doc_2_idx else -1}

def _get_num_pairs(rankings, num_neg_samples, bin_rankings=None):
  if bin_rankings:
    return reduce(lambda acc, ranking: acc + (len(ranking) - 1) * bin_rankings + bin_rankings * num_neg_samples if len(ranking) > bin_rankings else acc,
                  rankings,
                  0)
  else:
    return reduce(lambda acc, ranking: acc + (len(ranking[1]) ** 2) // 2 - len(ranking[1]) + len(ranking[1]) * num_neg_samples,
                  rankings,
                  0)

def _get_num_pos_pairs_with_bins(rankings, bin_rankings):
  return reduce(lambda acc, ranking: acc + (len(ranking) - 1) * bin_rankings if len(ranking) > bin_rankings else acc,
                rankings,
                0)

def _drop_next_n_from_ranking(num_to_drop_in_ranking, rankings):
  return [ranking[:1] + ranking[1 + num_to_drop_in_ranking:] for ranking in rankings]

class QueryPairwiseDataset(QueryDataset):
  def __init__(self,
               documents,
               data,
               rel_method=score,
               num_neg_samples=90,
               num_doc_tokens=100,
               rankings=None,
               query_tok_to_doc_tok=None,
               use_doc_out=False,
               bin_rankings=None,
               use_variable_loss=False,
               normalized_score_lookup=None,
               num_to_drop_in_ranking=0):
    if num_to_drop_in_ranking > 0:
      assert bin_rankings == 1, 'bin_rankings != 1 is not supported'
      rankings = _drop_next_n_from_ranking(num_to_drop_in_ranking, rankings)
    super().__init__(documents,
                     data,
                     rel_method=rel_method,
                     num_doc_tokens=num_doc_tokens,
                     rankings=rankings,
                     query_tok_to_doc_tok=query_tok_to_doc_tok)
    self.use_variable_loss = use_variable_loss
    self.bin_rankings = bin_rankings
    self.num_documents = len(documents)
    self.num_neg_samples = num_neg_samples
    self.rankings_for_train = self.rankings
    self.use_doc_out = use_doc_out
    self.normalized_score_lookup = normalized_score_lookup
    if bin_rankings:
      num_pairs_per_ranking = _.map_(self.rankings_for_train,
                                     lambda ranking: (len(ranking[1]) - 1) * bin_rankings if len(ranking) > bin_rankings else 0)
    else:
      num_pairs_per_ranking = _.map_(self.rankings_for_train,
                                     lambda ranking: (len(ranking[1]) ** 2) // 2 - len(ranking[1]))
    self.cumu_ranking_lengths = np.cumsum(num_pairs_per_ranking)
    self._num_pairs = None
    if self.bin_rankings:
      if self.bin_rankings != 1: raise NotImplementedError
      self._num_pos_pairs = _get_num_pos_pairs_with_bins(self.rankings_for_train,
                                                         self.bin_rankings)
    else:
      self._num_pos_pairs = _get_num_pairs(self.rankings_for_train, 0)

  def __len__(self):
    self._num_pairs = self._num_pairs or _get_num_pairs(self.rankings_for_train,
                                                        self.num_neg_samples,
                                                        self.bin_rankings)
    return self._num_pairs

  def __getitem__(self, idx):
    remapped_idx = idx % self._num_pos_pairs
    if idx >= self._num_pos_pairs:
      use_neg_sample = True
    else:
      use_neg_sample = False
    if self.bin_rankings:
      elem = _get_nth_pair_bin_rankings(self.rankings_for_train,
                                        self.cumu_ranking_lengths,
                                        self.bin_rankings,
                                        remapped_idx,
                                        self.use_variable_loss)
    else:
      elem = _get_nth_pair(self.rankings_for_train,
                           self.cumu_ranking_lengths,
                           remapped_idx,
                           self.use_variable_loss)
    order_int = elem['order_int']
    query = remap_if_exists(elem['query'], self.query_tok_to_doc_tok)
    doc_1 = self._get_document(elem['doc_id_1'])
    if use_neg_sample:
      doc_2 = self._get_document(choice(range(self.num_documents)))
      order_int = 1
    else:
      doc_2 = self._get_document(elem['doc_id_2'])
    if self.normalized_score_lookup is not None:
      doc_1_score = self.normalized_score_lookup[tuple(query)][elem['doc_id_1']]
      doc_2_score = self.normalized_score_lookup[tuple(query)][elem['doc_id_2']]
    else:
      doc_1_score = 0.0
      doc_2_score = 0.0
    return ((query, doc_1, doc_2, doc_1_score, doc_2_score), order_int)

def score_documents_embed(doc_word_embeds, query_word_embeds, documents, queries, device):
  query_embeds = query_word_embeds(queries)
  query_vecs = query_embeds.sum(1).to(device)
  query_vecs = query_vecs / (torch.norm(query_vecs, 2, 1).unsqueeze(1) + 0.0001)
  doc_embeds = doc_word_embeds(documents)
  doc_vecs = doc_embeds.sum(1).to(device)
  doc_vecs = doc_vecs / (torch.norm(doc_vecs, 2, 1).unsqueeze(1) + 0.0001)
  scores = torch.zeros(len(query_vecs), len(doc_vecs), device=device)
  sections = torch.cat([torch.arange(start=0, end=len(doc_vecs), step=1000, dtype=torch.long),
                        torch.tensor([len(doc_vecs)], dtype=torch.long)])
  for chunk_start, chunk_end in zip(sections, sections[1:]):
    logits = (doc_vecs[chunk_start:chunk_end] * query_vecs.unsqueeze(1)).sum(2)
    scores[:, chunk_start:chunk_end] = logits
  return scores

def get_top_k(scores, k=1000):
  sorted_scores, idxs = torch.sort(scores, descending=True)
  return idxs[:k]

class TrueRandomSampler(Sampler):
  def __init__(self, data_source):
    self.data_source = data_source
    self.num_samples_seen = 0

  def __iter__(self):
    while self.num_samples_seen < len(self.data_source):
      yield randint(0, len(self.data_source) - 1)
      self.num_samples_seen += 1
    self.num_samples_seen = 0

  def __len__(self):
    return len(self.data_source)

def build_query_dataloader(documents,
                           normalized_data,
                           batch_size,
                           rel_method=score,
                           cache=None,
                           num_doc_tokens=100,
                           limit=None,
                           query_tok_to_doc_tok=None,
                           use_sequential_sampler=False,
                           use_doc_out=False,
                           normalized_score_lookup=None) -> DataLoader:
  rankings = read_cache(cache, lambda: to_query_rankings_pairs(normalized_data, limit=limit)) if cache is not None else None
  dataset = QueryDataset(documents,
                         normalized_data,
                         rel_method=rel_method,
                         rankings=rankings,
                         num_doc_tokens=num_doc_tokens,
                         query_tok_to_doc_tok=query_tok_to_doc_tok,
                         use_doc_out=use_doc_out,
                         normalized_score_lookup=normalized_score_lookup)
  sampler = SequentialSampler if use_sequential_sampler else TrueRandomSampler
  return DataLoader(dataset,
                    batch_sampler=BatchSampler(sampler(dataset), batch_size, False),
                    collate_fn=collate_query_samples)

def build_query_pairwise_dataloader(documents,
                                    data,
                                    batch_size,
                                    rel_method=score,
                                    num_neg_samples=90,
                                    cache=None,
                                    num_doc_tokens=100,
                                    limit=None,
                                    query_tok_to_doc_tok=None,
                                    use_sequential_sampler=False,
                                    use_doc_out=False,
                                    bin_rankings=None,
                                    use_variable_loss=False,
                                    normalized_score_lookup=None,
                                    num_to_drop_in_ranking=0) -> DataLoader:
  rankings = read_cache(cache, lambda: to_query_rankings_pairs(data, limit=limit)) if cache is not None else None
  dataset = QueryPairwiseDataset(documents,
                                 data,
                                 rel_method=rel_method,
                                 num_neg_samples=num_neg_samples,
                                 rankings=rankings,
                                 num_doc_tokens=num_doc_tokens,
                                 query_tok_to_doc_tok=query_tok_to_doc_tok,
                                 use_doc_out=use_doc_out,
                                 bin_rankings=bin_rankings,
                                 use_variable_loss=use_variable_loss,
                                 normalized_score_lookup=normalized_score_lookup,
                                 num_to_drop_in_ranking=num_to_drop_in_ranking)
  sampler = SequentialSampler if use_sequential_sampler else TrueRandomSampler
  return DataLoader(dataset,
                    batch_sampler=BatchSampler(sampler(dataset), batch_size, False),
                    collate_fn=collate_query_pairwise_samples)
