import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler

import scipy.sparse as sp
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer

from preprocessing import collate_query_samples, to_query_rankings_pairs, pad_to_max_len, all_ones, score, inv_log_rank, inv_rank, exp_score

class QueryDataset(Dataset):
  def __init__(self, documents, data, rel_method=score):
    self.documents = documents
    self.data = data
    self.rel_method = rel_method

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return ((self.data[idx]['query'], self.documents[self.data[idx]['document_id']]),
            self.rel_method(self.data[idx]))

def _get_tfidf_transformer_and_matrix(documents):
  transformer = TfidfTransformer()
  counts = sp.lil_matrix((len(documents), documents.max().item() + 1))
  for doc_num, doc in enumerate(documents):
    doc_counts = np.bincount(doc)
    nonzero = doc_counts.nonzero()
    counts[doc_num, nonzero] = doc_counts[nonzero]
  return transformer, transformer.fit_transform(counts)

def _get_top_scoring(tfidf_docs, query, k=1000):
  subset = tfidf_docs[:, query]
  scores = torch.tensor(subset.sum(1).T.tolist()).squeeze()
  sorted_scores, idxs = torch.sort(scores, descending=True)
  return idxs[:k]

class RankingDataset(Dataset):
  def __init__(self, documents, data, k=10):
    self.documents = torch.tensor(pad_to_max_len(documents), dtype=torch.long)
    self.k = k
    self.rankings = to_query_rankings_pairs(data, k=self.k)
    self.tfidf_transformer, self.tfidf_docs = _get_tfidf_transformer_and_matrix(self.documents)

  def __len__(self):
    return len(self.rankings)

  def __getitem__(self, idx):
    query, ranking = self.rankings[idx]
    ranking = torch.tensor(ranking, dtype=torch.long)
    return {'query': torch.tensor(query, dtype=torch.long),
            'documents': self.documents[_get_top_scoring(self.tfidf_docs, query)],
            'ranking': ranking,
            'relevant': ranking}

def build_query_dataloader(documents, data, rel_method=score) -> DataLoader:
  dataset = QueryDataset(documents, data, rel_method=rel_method)
  return DataLoader(dataset,
                    batch_sampler=BatchSampler(RandomSampler(dataset), 100, False),
                    collate_fn=collate_query_samples)
