import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.sgdr import Callback

from pointwise_ranker import PointwiseRanker


class MetricRecorder(Callback):
  '''
  Prints a metric value which requires external information
  '''
  def __init__(self, model):
    super().__init__()
    self.model = model

  def on_train_begin(self):
    self.iterations, self.epochs = [],[]
    self.rec_metrics = []
    self.iteration = 0
    self.epoch = 0

  def on_epoch_end(self, other_metrics):
    self.epoch += 1
    self.epochs.append(self.iteration)

  def on_batch_end(self, validation_loss):
    self.iteration += 1
    self.iterations.append(self.iteration)

  def save_metrics(self, other_metrics):
    pass

  def plot_loss(self, n_skip=10, n_skip_end=5):
    pass

  def plot_lr(self):
    pass

class RankingMetricRecorder(MetricRecorder):
  def __init__(self, model, train_ranking_dl, test_ranking_dl):
    super().__init__(model)
    self.ranker = PointwiseRanker(model)
    self.train_ranking_dl = train_ranking_dl
    self.test_ranking_dl = test_ranking_dl

  def in_top_k(self, query, documents, relevant_document_index, k=10):
    top_k = self.ranker(query, documents)[:, :k]
    acc = 0
    for i, doc in enumerate(relevant_document_index):
      if int(doc) in top_k[i]: acc += 1
    return acc / k

  def metrics_at_k(self, dataset, k=10):
    correct = 0
    num_relevant = 0
    num_rankings_considered = 0
    for to_rank in dataset:
      if num_rankings_considered > 10: break
      assert len(to_rank['documents']) >= k, "specified k is greater than the number of documents to rank"
      ranking = torch.squeeze(self.ranker(torch.unsqueeze(to_rank['query'], 0),
                                          to_rank['documents']))
      for doc_id in ranking[:k].tolist():
        correct += doc_id in to_rank['relevant']
      num_relevant += len(to_rank['relevant'])
      num_rankings_considered += 1
    precision_k = correct / (k * num_rankings_considered)
    recall_k = correct / num_relevant
    return precision_k, recall_k
o
  def on_epoch_end(self, other_metrics):
    self.epoch += 1
    self.epochs.append(self.iteration)
    print(self.metrics_at_k(self.train_ranking_dl))
    print(self.metrics_at_k(self.test_ranking_dl))


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
