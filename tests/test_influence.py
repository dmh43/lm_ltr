# pylint: disable=redefined-outer-name
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
import torch.optim as optim

import lm_ltr.influence as i

NUM_FEATURES = 10
BATCH_SIZE = 512
NUM_TRAIN_SAMPLES = 10000

class SimpleModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.lin = nn.Linear(NUM_FEATURES, 1)

  def forward(self, x):
    return self.lin(x).squeeze()

@pytest.fixture(scope='module')
def criterion(): return nn.BCEWithLogitsLoss()

@pytest.fixture(scope='module')
def train_dataloader():
  num_samples = NUM_TRAIN_SAMPLES
  xs = torch.randn((num_samples, NUM_FEATURES)) + torch.arange(num_samples).unsqueeze(1).float()
  targets = (torch.arange(num_samples) > num_samples / 2).float()
  targets[-100:] = 1 - targets[-100:]
  targets[:100] = 1 - targets[:100]
  dataset = TensorDataset(xs, targets)
  batch_sampler = BatchSampler(RandomSampler(dataset), BATCH_SIZE, False)
  return DataLoader(dataset, batch_sampler=batch_sampler)

@pytest.fixture(scope='module')
def test_dataset():
  num_samples = 100
  xs = torch.arange(0, NUM_TRAIN_SAMPLES, 100).repeat(NUM_FEATURES, 1).t().float()
  targets = (xs[:, 0] > NUM_TRAIN_SAMPLES / 2).float()
  return TensorDataset(xs, targets)

@pytest.fixture(scope='module')
def trained_model(criterion, train_dataloader):
  model = SimpleModel()
  optimizer = optim.Adam(model.parameters())
  for epoch_num in range(20):
    for xs, targets in train_dataloader:
      optimizer.zero_grad()
      loss = criterion(model(xs), targets)
      loss.backward()
      optimizer.step()
  print('loss:', loss)
  model.eval()
  return model

def test_calc_influence(criterion, trained_model, train_dataloader, test_dataset):
  collate_fn = lambda sample: [((sample[0][0].unsqueeze(0),), sample[0][1])]
  test_hvps = i.calc_test_hvps(criterion,
                               trained_model,
                               train_dataloader,
                               test_dataset,
                               collate_fn)
  influences = []
  for train_sample in train_dataloader.dataset:
    influences.append(i.calc_influence(criterion, trained_model, train_sample, test_hvps).sum())
  influences = torch.tensor(influences)
  largest_vals, largest_idxs = torch.topk(influences, k=100)
  most_neg_vals, most_neg_idxs = torch.topk(-influences, k=100)
  assert all((train_dataloader.dataset[idx][1] == 1) == (train_dataloader.dataset[idx][0][0] > NUM_TRAIN_SAMPLES/2)
             for idx in largest_idxs)
  assert all((train_dataloader.dataset[idx][1] == 0) == (train_dataloader.dataset[idx][0][0] <= NUM_TRAIN_SAMPLES/2)
             for idx in largest_idxs)
  assert all((train_dataloader.dataset[idx][1] == 0) == (train_dataloader.dataset[idx][0][0] > NUM_TRAIN_SAMPLES/2)
             for idx in most_neg_idxs)
  assert all((train_dataloader.dataset[idx][1] == 1) == (train_dataloader.dataset[idx][0][0] <= NUM_TRAIN_SAMPLES/2)
             for idx in most_neg_idxs)
