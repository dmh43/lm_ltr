import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler

from preprocessing import collate_query_samples, to_query_rankings_pairs, pad_to_max_len

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

class RankingDataset(Dataset):
  def __init__(self, documents, data, k=10):
    self.documents = torch.tensor(pad_to_max_len(documents), dtype=torch.long)
    self.k = k
    self.rankings = to_query_rankings_pairs(data, k=self.k)

  def __len__(self):
    return len(self.rankings)

  def __getitem__(self, idx):
    query, ranking = self.rankings[idx]
    ranking = torch.tensor(ranking, dtype=torch.long)
    return {'query': torch.tensor(query, dtype=torch.long),
            'documents': self.documents,
            'ranking': ranking,
            'relevant': ranking}

def build_query_dataloader(documents, data, rel_method=score) -> DataLoader:
  dataset = QueryDataset(documents, data, rel_method=rel_method)
  return DataLoader(dataset,
                    batch_sampler=BatchSampler(RandomSampler(dataset), 100, False),
                    collate_fn=collate_query_samples)
