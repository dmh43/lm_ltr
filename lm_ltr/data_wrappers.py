from typing import List

from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.sampler import SequentialSampler, BatchSampler

from preprocessing import collate

class QueryDataset(Dataset):
  def __init__(self, documents: List[int], queries: List[int], document_ids: List[int], labels: List[int]) -> None:
    assert len(queries) == len(labels)
    self.labels = labels
    self.queries = queries
    self.document_ids = document_ids
    self.documents = documents

  def __len__(self):
    return len(self.queries)

  def __getitem__(self, idx):
    return ((self.queries[idx], self.documents[self.document_ids[idx]]), self.labels[idx])

def build_dataloader(documents: List[int], queries: List[int], document_ids: List[int], labels: List[int]) -> DataLoader:
  dataset = QueryDataset(documents, queries, document_ids, labels)
  return DataLoader(dataset,
                    # batch_sampler=BatchSampler(RandomSampler(dataset), 100, False),
                    batch_sampler=BatchSampler(SequentialSampler(dataset), 100, False),
                    collate_fn=collate)
