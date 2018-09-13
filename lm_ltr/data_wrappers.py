import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler

from preprocessing import collate, to_query_rankings_pairs

class QueryDataset(Dataset):
  def __init__(self, documents, data):
    self.documents = documents
    self.data = data
    self.num_negative_samples = 0
    self.relevant_docs = {}
    for row in self.data:
      query = row['query']
      if str(query) in self.relevant_docs:
        self.relevant_docs[str(query)].append(row['document_id'])
      else:
        self.relevant_docs[str(query)] = [row['document_id']]
    self.idxs = list(range(len(self.documents)))

  def __len__(self):
    return len(self.data) + len(self.data) * self.num_negative_samples

  def __getitem__(self, idx):
    if idx >= len(self.data):
      idx_to_use = idx % len(self.data)
      doc_id = self.data[idx_to_use]['document_id']
      while doc_id in self.relevant_docs[str(self.data[idx_to_use]['query'])]:
        doc_id = random.choice(self.idxs)
      return ((self.data[idx_to_use]['query'], self.documents[doc_id]), 0.0)
    else:
      return ((self.data[idx]['query'], self.documents[self.data[idx]['document_id']]), self.data[idx]['rel'])

class RankingDataset(Dataset):
  def __init__(self, documents, data):
    self.documents = documents
    self.rankings = to_query_rankings_pairs(data)

  def __len__(self):
    return len(self.rankings)

  def __getitem__(self, idx):
    query, ranking = self.rankings[idx]
    return query, ranking

def build_query_dataloader(documents, data) -> DataLoader:
  dataset = QueryDataset(documents, data)
  return DataLoader(dataset,
                    batch_sampler=BatchSampler(RandomSampler(dataset), 1000, False),
                    collate_fn=collate)

def build_ranking_dataloader(documents, data) -> DataLoader:
  dataset = RankingDataset(documents, data)
  return DataLoader(dataset,
                    batch_sampler=BatchSampler(RandomSampler(dataset), 1000, False),
                    collate_fn=collate)
