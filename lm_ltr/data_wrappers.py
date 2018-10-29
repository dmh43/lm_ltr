import ast
from random import sample, randint
from functools import reduce
import pydash as _

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, Sampler

import scipy.sparse as sp
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer

from .preprocessing import collate_query_samples, collate_query_pairwise_samples, to_query_rankings_pairs, pad_to_max_len, all_ones, score, inv_log_rank, inv_rank, exp_score
from .utils import append_at

class QueryDataset(Dataset):
  def __init__(self, documents, data, rel_method=score, num_doc_tokens=100):
    self.documents = documents
    self.data = data
    self.rel_method = rel_method
    self.rankings = to_query_rankings_pairs(data)
    self.num_doc_tokens = num_doc_tokens

  def _get_document(self, elem_idx):
    return self.documents[self.data[elem_idx]['doc_id']][:self.num_doc_tokens]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return ((self.data[idx]['query'], self._get_document(idx)),
            self.rel_method(self.data[idx]))

class RankingDataset(Dataset):
  def __init__(self,
               documents,
               rankings,
               relevant=None,
               num_doc_tokens=100,
               k=10):
    self.rankings = rankings
    self.documents = documents
    self.k = k
    self.num_doc_tokens = num_doc_tokens
    self.num_to_rank = 1000
    self.is_test = relevant is not None
    self.relevant = relevant

  def __len__(self):
    return len(self.rankings)

  def _get_train_item(self, idx):
    query, ranking = self.rankings[idx]
    relevant = set(ranking[:self.k])
    if len(ranking) < self.num_to_rank:
      neg_samples = sample(set(range(self.num_to_rank)) - set(ranking),
                           self.num_to_rank - len(ranking))
      ranking_with_neg = ranking + neg_samples
    else:
      ranking_with_neg = ranking
    return {'query': torch.tensor(query, dtype=torch.long),
            'documents': [self.documents[idx] for idx in ranking_with_neg],
            'doc_ids': torch.tensor(ranking_with_neg, dtype=torch.long),
            'ranking': ranking[:self.k],
            'relevant': relevant}

  def _get_test_item(self, idx):
    query, ranking = self.rankings[idx]
    query, relevant = self.relevant[idx]
    relevant = set(relevant)
    return {'query': torch.tensor(query, dtype=torch.long),
            'documents': [self.documents[idx] for idx in ranking],
            'doc_ids': torch.tensor(ranking, dtype=torch.long),
            'ranking': self.relevant[idx][:self.k],
            'relevant': relevant}

  def __getitem__(self, idx):
    return self._get_test_item(idx) if self.is_test else self._get_train_item(idx)

def _get_nth_pair(rankings, cumu_num_pairs, idx):
  ranking_idx = np.searchsorted(cumu_num_pairs, idx, side='right')
  offset = idx - cumu_num_pairs[ranking_idx - 1] if ranking_idx != 0 else idx
  query = rankings[ranking_idx][0]
  doc_ids = rankings[ranking_idx][1]
  num_doc_ids = len(doc_ids)
  doc_1_idx = offset // (num_doc_ids - 1)
  doc_2_idx = offset % (num_doc_ids - 1)
  if doc_2_idx >= doc_1_idx:
    doc_2_idx += 1
  return {'query': query,
          'doc_id_1': doc_ids[doc_1_idx],
          'doc_id_2': doc_ids[doc_2_idx],
          'order_int': 1 if doc_1_idx < doc_2_idx else -1}

def _get_num_pairs(rankings):
  return reduce(lambda acc, ranking: acc + len(ranking[1]) ** 2 - len(ranking[1]),
                rankings,
                0)

def insert_negative_samples(num_documents, num_neg_samples, rankings):
  for query, ranking in rankings:
    ranking.extend(sample(range(num_documents), num_neg_samples))

class QueryPairwiseDataset(QueryDataset):
  def __init__(self, documents, data, rel_method=score, num_neg_samples=90):
    super().__init__(documents, data)
    num_documents = len(documents)
    self.num_neg_samples = num_neg_samples
    insert_negative_samples(num_documents, self.num_neg_samples, self.rankings)
    self.rankings_for_train = self.rankings
    num_pairs_per_ranking = _.map_(self.rankings_for_train,
                                   lambda ranking: len(ranking[1]) ** 2 - len(ranking[1]))
    self.cumu_ranking_lengths = np.cumsum(num_pairs_per_ranking)
    self._num_pairs = None

  def __len__(self):
    self._num_pairs = self._num_pairs or _get_num_pairs(self.rankings_for_train)
    return self._num_pairs

  def __getitem__(self, idx):
    elem = _get_nth_pair(self.rankings_for_train, self.cumu_ranking_lengths, idx)
    return ((elem['query'],
             self.documents[elem['doc_id_1']][:self.num_doc_tokens],
             self.documents[elem['doc_id_2']][:self.num_doc_tokens]),
            elem['order_int'])

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

def normalize_scores_query_wise(data):
  query_doc_info = {}
  for row in data:
    score = row.get('score') or 0.0
    append_at(query_doc_info, str(row['query'])[1:-1], [row['doc_id'], score, row['query']])
  normalized_data = []
  for doc_infos in query_doc_info.values():
    scores = torch.tensor([doc_score for doc_id, doc_score, query in doc_infos], device=torch.device('cuda'))
    query_score_total = torch.logsumexp(scores, 0)
    normalized_scores = scores - query_score_total
    normalized_data.extend([{'query': doc_info[2],
                             'doc_id': doc_info[0],
                             'score': score}
                            for doc_info, score in zip(doc_infos, scores.tolist())])
  return normalized_data

class TrueRandomSampler(Sampler):
  def __init__(self, data_source):
    self.data_source = data_source
    self.num_samples_seen = 0

  def __iter__(self):
    while self.num_samples_seen < len(self.data_source):
      yield randint(0, len(self.data_source) - 1)
      self.num_samples_seen += 1

  def __len__(self):
    return len(self.data_source)

def build_query_dataloader(documents, data, batch_size, rel_method=score) -> DataLoader:
  normalized_data = normalize_scores_query_wise(data)
  dataset = QueryDataset(documents, normalized_data, rel_method=rel_method)
  return DataLoader(dataset,
                    batch_sampler=BatchSampler(TrueRandomSampler(dataset), batch_size, False),
                    collate_fn=collate_query_samples)

def build_query_pairwise_dataloader(documents, data, batch_size, rel_method=score, num_neg_samples=90) -> DataLoader:
  dataset = QueryPairwiseDataset(documents, data, rel_method=rel_method, num_neg_samples=num_neg_samples)
  return DataLoader(dataset,
                    batch_sampler=BatchSampler(TrueRandomSampler(dataset), batch_size, False),
                    collate_fn=collate_query_pairwise_samples)
