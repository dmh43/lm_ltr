from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai import Callback
import numpy as np

from .pointwise_ranker import PointwiseRanker


class MetricRecorder(Callback):
  '''
  Prints a metric value which requires external information
  '''
  def __init__(self, model):
    super().__init__()
    self.model = model

class RankingMetricRecorder(MetricRecorder):
  def __init__(self, device, model, train_ranking_dl, test_ranking_dl):
    super().__init__(model)
    self.device = device
    self.ranker = PointwiseRanker(device, model)
    self.train_ranking_dl = train_ranking_dl
    self.test_ranking_dl = test_ranking_dl
    self.log = open('./results', 'a+')
    self.log.write(f'New train run at: {time()}\n')

  def metrics_at_k(self, dataset, k=10):
    correct = 0
    num_relevant = 0
    num_rankings_considered = 0
    dcg = 0
    idcg = 0
    for to_rank in dataset:
      if num_rankings_considered > 100: break
      if len(to_rank['documents']) < k: continue
      ranking_ids_for_batch = torch.squeeze(self.ranker(torch.unsqueeze(to_rank['query'], 0),
                                                        to_rank['documents']))
      ranking = to_rank['doc_ids'][ranking_ids_for_batch]
      for doc_rank, doc_id in enumerate(ranking[:k].tolist()):
        rel = doc_id in to_rank['relevant']
        correct += rel
        dcg += (2 ** rel - 1) / np.log2(doc_rank + 2)
      num_relevant += len(to_rank['relevant'])
      idcg += np.array([1.0/np.log2(rank + 2) for rank in range(min(k, len(to_rank['relevant'])))]).sum()
      if len(to_rank['relevant']) > 0:
        num_rankings_considered += 1
    precision_k = correct / (k * num_rankings_considered)
    recall_k = correct / num_relevant
    ndcg = dcg / idcg
    return precision_k, recall_k, ndcg

  def on_epoch_begin(self, **kwargs):
    train_results = self.metrics_at_k(self.train_ranking_dl)
    test_results = self.metrics_at_k(self.test_ranking_dl)
    report = '\n'.join(['Train: ' + str(train_results),
                        'Test: ' + str(test_results)])
    self.log.write(report)
    print(report)

  def on_train_end(self, **kwargs):
    self.log.close()


def recall(logits, targs, thresh=0.5, epsilon=1e-8):
  preds = F.sigmoid(logits) > thresh
  tpos = torch.mul((targs.byte() == preds.byte()), targs.byte()).float()
  return tpos.sum()/(targs.sum() + epsilon)

def precision(logits, targs, thresh=0.5, epsilon=1e-8):
  preds = (F.sigmoid(logits) > thresh).float()
  tpos = torch.mul((targs.byte() == preds.byte()), targs.byte()).float()
  return tpos.sum()/(preds.sum() + epsilon)

def f1(logits, targs, thresh=0.5, epsilon=1e-8):
  rec = recall(logits, targs, thresh)
  prec = precision(logits, targs, thresh)
  return 2 * prec * rec / (prec + rec + epsilon)
