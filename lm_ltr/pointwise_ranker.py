import torch
import torch.nn as nn

class PointwiseRanker:
  def __init__(self, pointwise_scorer):
    self.pointwise_scorer = pointwise_scorer

  def __call__(self, query, documents):
    assert len(query.shape) == 2, "PointwiseRanker expects a single batch of queries"
    ranks = []
    for query in query:
      scores = self.pointwise_scorer(torch.unsqueeze(query, 0).repeat(len(documents), 1),
                                     documents)
      _, sort_idx = torch.sort(scores, descending=True)
      ranks.append(sort_idx)
    return torch.stack(ranks)
