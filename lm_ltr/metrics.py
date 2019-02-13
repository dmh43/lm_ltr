from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai import Callback
import numpy as np
import pydash as _

from .pointwise_ranker import PointwiseRanker
from .utils import at_least_one_dim, to_list


class MetricRecorder(Callback):
  '''
  Prints a metric value which requires external information
  '''
  def __init__(self, model):
    super().__init__()
    self.model = model

class RankingMetricRecorder(MetricRecorder):
  def __init__(self,
               device,
               model,
               train_ranking_dataset,
               val_ranking_dataset,
               test_ranking_dataset,
               experiment,
               doc_chunk_size=-1,
               dont_smooth=False,
               dont_include_normalized_score=False,
               use_dense=False,
               record_every_n=10000):
    super().__init__(model)
    self.device = device
    self.ranker = PointwiseRanker(device,
                                  self.model,
                                  doc_chunk_size,
                                  use_doc_scores_for_smoothing=not dont_smooth,
                                  dont_include_normalized_score=dont_include_normalized_score,
                                  use_dense=use_dense)
    self.train_ranking_dataset = train_ranking_dataset
    self.val_ranking_dataset = val_ranking_dataset
    self.test_ranking_dataset = test_ranking_dataset
    self.experiment_context = None
    self.experiment = experiment
    self.dont_smooth = dont_smooth
    self.record_every_n = record_every_n

  def metrics_at_k(self, dataset, smooth, k=10):
    relevant_doc_ids = (to_rank['relevant'] for to_rank in dataset)
    def rank_dataset_contents():
      with torch.no_grad():
        num_rankings_considered = 0
        for to_rank in dataset:
          if num_rankings_considered > 300: break
          if len(to_rank['documents']) < k:
            yield None
            continue
          ranking_ids_for_batch = torch.squeeze(self.ranker(torch.unsqueeze(to_rank['query'], 0),
                                                            to_rank['documents'],
                                                            to_rank['doc_scores'],
                                                            smooth=smooth,
                                                            k=k))
          ranking = to_rank['doc_ids'][ranking_ids_for_batch]
          yield at_least_one_dim(ranking)
          num_rankings_considered += 1
    return metrics_at_k(rank_dataset_contents(), relevant_doc_ids, k=k)

  def _find_best_smooth(self, inc=0.01, metric='ndcg'):
    best_smooth = None
    best_metric_val = None
    val_best_metrics = None
    for smooth in np.arange(0, 1, inc):
      val_metrics = self.metrics_at_k(self.val_ranking_dataset, smooth)
      smooth_metric_val = val_metrics[metric]
      if (best_smooth is None) or (smooth_metric_val > best_metric_val):
        best_metric_val = smooth_metric_val
        best_smooth = smooth
        val_best_metrics = val_metrics
    return best_smooth, val_best_metrics

  def _check(self, batch_num=0):
    if self.dont_smooth:
      smooth = 0.0
      val_results = self.metrics_at_k(self.val_ranking_dataset, smooth)
    else:
      smooth, val_results = self._find_best_smooth()
    train_results = self.metrics_at_k(self.train_ranking_dataset, smooth)
    test_results = self.metrics_at_k(self.test_ranking_dataset, smooth)
    test_results_no_smooth = self.metrics_at_k(self.test_ranking_dataset, 0.0)
    self.experiment.record_metrics(_.assign({},
                                            _.map_keys(train_results, lambda val, key: 'train_' + key),
                                            _.map_keys(test_results, lambda val, key: 'test_' + key),
                                            _.map_keys(test_results_no_smooth, lambda val, key: 'test_no_smooth_' + key),
                                            _.map_keys(val_results, lambda val, key: 'val_' + key)),
                                   batch_num)

  def on_batch_begin(self, num_batch, **kwargs):
    if num_batch % self.record_every_n == 0:
      self._check(num_batch)

  def on_epoch_begin(self, epoch, **kwargs):
    self.experiment.update_epoch(epoch)

  def on_epoch_end(self, num_batch, **kwargs):
    self._check(num_batch)

  def on_train_begin(self, **kwargs):
    self.experiment_context = self.experiment.train(['train_precision',
                                                     'train_recall',
                                                     'train_ndcg',
                                                     'train_map',
                                                     'test_precision',
                                                     'test_recall',
                                                     'test_ndcg',
                                                     'test_map',
                                                     'test_no_smooth_precision',
                                                     'test_no_smooth_recall',
                                                     'test_no_smooth_ndcg',
                                                     'test_no_smooth_map',
                                                     'val_precision',
                                                     'val_recall',
                                                     'val_ndcg',
                                                     'val_map'])
    self.experiment_context.__enter__()

  def on_train_end(self, num_batch, **kwargs):
    self.experiment_context.__exit__()

def recall(logits, targs, thresh=0.5, epsilon=1e-8):
  preds = torch.sigmoid(logits) > thresh
  tpos = torch.mul((targs.byte() == preds.byte()), targs.byte()).float()
  return tpos.sum()/(targs.sum() + epsilon)

def precision(logits, targs, thresh=0.5, epsilon=1e-8):
  preds = (torch.sigmoid(logits) > thresh).float()
  tpos = torch.mul((targs.byte() == preds.byte()), targs.byte()).float()
  return tpos.sum()/(preds.sum() + epsilon)

def f1(logits, targs, thresh=0.5, epsilon=1e-8):
  rec = recall(logits, targs, thresh)
  prec = precision(logits, targs, thresh)
  return 2 * prec * rec / (prec + rec + epsilon)

def metrics_at_k(rankings_to_judge, relevant_doc_ids, k=10):
  correct = 0
  num_relevant = 0
  num_rankings_considered = 0
  avg_precision_sum = 0
  ndcgs = []
  for ranking, relevant in zip(rankings_to_judge, relevant_doc_ids):
    if ranking is None: continue
    num_relevant_in_ranking = len(relevant)
    if num_relevant_in_ranking == 0: continue
    avg_correct = 0
    correct_in_ranking = 0
    dcg = 0
    idcg = 0
    for doc_rank, doc_id in enumerate(to_list(ranking)[:k]):
      rel = doc_id in relevant
      correct += rel
      correct_in_ranking += rel
      precision_so_far = correct_in_ranking / (doc_rank + 1)
      avg_correct += rel * precision_so_far
      dcg += (2 ** rel - 1) / np.log2(doc_rank + 2)
    num_relevant += num_relevant_in_ranking
    avg_precision_sum += avg_correct / min(k, num_relevant_in_ranking)
    idcg += np.array([1.0/np.log2(rank + 2)
                      for rank in range(min(k, num_relevant_in_ranking))]).sum()
    ndcgs.append(dcg / idcg)
    num_rankings_considered += 1
  precision_k = correct / (k * num_rankings_considered)
  recall_k = correct / num_relevant
  ndcg = sum(ndcgs) / len(ndcgs)
  mean_avg_precision = avg_precision_sum / num_rankings_considered
  return {'precision': precision_k,
          'recall': recall_k,
          'ndcg': ndcg,
          'map': mean_avg_precision}
