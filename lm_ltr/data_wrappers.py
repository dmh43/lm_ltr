from typing import List

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler

from preprocessing import collate

class QueryDataset(Dataset):
  def __init__(self, documents: List[int], queries: List[int], labels: List[List[int]]) -> None:
    assert len(queries) == len(labels)
    self.labels = labels
    self.queries = queries
    self.documents = documents

  def __len__(self):
    return len(self.queries)

  def __getitem__(self, idx):
    candidate_documents = range(len(self.documents))
    return ((self.queries[idx], candidate_documents), self.labels[idx])

def build_dataloader(documents: List[int], queries: List[int], labels: List[int]) -> DataLoader:
  dataset = QueryDataset(documents, queries, labels)
  return DataLoader(dataset,
                    batch_sampler=BatchSampler(RandomSampler(dataset), 100, False),
                    collate_fn=collate)
