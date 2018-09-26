import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler

from preprocessing import collate_query_samples, collate_term_matching_samples, to_query_rankings_pairs, pad_to_max_len, get_term_matching

class QueryDataset(Dataset):
  def __init__(self, documents, data):
    self.documents = documents
    self.data = data
    self.idxs = list(range(len(self.documents)))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return ((self.data[idx]['query'], self.documents[self.data[idx]['document_id']]),
            self.data[idx]['rel'])

class TermMatchingDataset(QueryDataset):
  def __init__(self, query_document_token_mapping, documents, data):
    super().__init__(documents, data)
    self.query_document_token_mapping = query_document_token_mapping

  def __getitem__(self, idx):
    x, y = super().__getitem__(idx)
    return (get_term_matching(self.query_document_token_mapping,
                              *x),
            y)

class RankingDataset(Dataset):
  def __init__(self, documents, data):
    self.documents = torch.tensor(pad_to_max_len(documents), dtype=torch.long)
    self.rankings = to_query_rankings_pairs(data)

  def __len__(self):
    return len(self.rankings)

  def __getitem__(self, idx):
    query, ranking = self.rankings[idx]
    return {'query': torch.tensor(query, dtype=torch.long),
            'documents': self.documents,
            'ranking': torch.tensor(ranking, dtype=torch.long),
            'relevant': ranking}

def build_query_dataloader(documents, data) -> DataLoader:
  dataset = QueryDataset(documents, data)
  return DataLoader(dataset,
                    batch_sampler=BatchSampler(RandomSampler(dataset), 100, False),
                    collate_fn=collate_query_samples)

def build_term_matching_dataloader(query_document_token_mapping, documents, data):
  dataset = TermMatchingDataset(query_document_token_mapping, documents, data)
  return DataLoader(dataset,
                    batch_sampler=BatchSampler(RandomSampler(dataset), 100, False),
                    collate_fn=collate_term_matching_samples)
