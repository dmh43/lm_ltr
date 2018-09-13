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
  def __init__(self, model, all_documents, train_ranking_dl, test_ranking_dl):
    super().__init__(model)
    self.ranker = PointwiseRanker(model, True)
    self.train_ranking_dl = train_ranking_dl
    self.test_ranking_dl = test_ranking_dl
    self.all_documents = all_documents

  def in_top_k(self, query, documents, relevant_document_index):
    top_k = self.ranker(query, documents)[:, :self.k]
    acc = 0
    for i, doc in enumerate(relevant_document_index):
      if int(doc) in top_k[i]: acc += 1
    return acc / self.k

  def precision_at_k(dataset, k=10)

  def on_epoch_end(self, other_metrics):
    self.epoch += 1
    self.epochs.append(self.iteration)
    if len(self.train_data[1].shape) == 3:
      candidate_documents = self.train_data[1]
    else:
      candidate_documents = self.all_documents
    print(self.precision_at_k(self.train_dl))
    print(self.precision_at_k(self.test_dl))


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
