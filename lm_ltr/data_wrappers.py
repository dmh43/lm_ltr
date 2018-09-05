from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler

from preprocessing import collate

class QueryDataset(Dataset):
  def __init__(self, documents, data):
    self.documents = documents
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return ((self.data[idx]['query'], self.documents[self.data[idx]['document_id']]), self.data[idx]['rel'])

def build_dataloader(documents, data) -> DataLoader:
  dataset = QueryDataset(documents, data)
  return DataLoader(dataset,
                    batch_sampler=BatchSampler(RandomSampler(dataset), 100, False),
                    collate_fn=collate)
