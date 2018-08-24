from typing import List

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler
from fastai.text import Tokenizer

def tokens_to_indexes(tokens: List[List[str]]):
  lookup: dict = {}
  result = []
  for tokens_chunk in tokens:
    chunk_result = []
    for token in tokens_chunk:
      lookup[token] = lookup.get(token) or len(lookup)
      chunk_result.append(lookup[token])
    result.append(chunk_result)
  return result, lookup

class QueryDataset(Dataset):
  def __init__(self, documents: List[str], queries: List[str], labels: List[int]) -> None:
    assert len(queries) == len(labels)
    self.labels = labels
    self.queries = queries
    self.documents = documents
    self.tokenizer = Tokenizer()
    self.tokenized_queries = [self.tokenizer.proc_text(q) for q in self.queries]
    self.idx_queries, self.query_term_lookup = tokens_to_indexes(self.tokenized_queries)
    self.tokenized_documents = [self.tokenizer.proc_text(doc) for doc in self.documents]
    self.idx_documents, self.document_term_lookup = tokens_to_indexes(self.tokenized_documents)

  def __len__(self):
    return len(self.queries)

  def __getitem__(self, idx):
    return {'query': self.queries[idx], 'label': self.labels[idx]}

def build_dataloader(documents: List[str], queries: List[str], labels: List[int]) -> DataLoader:
  dataset = QueryDataset(documents, queries, labels)
  return DataLoader(dataset, batch_sampler=BatchSampler(RandomSampler(dataset), 100, False))
