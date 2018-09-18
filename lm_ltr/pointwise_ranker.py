import torch
import torch.nn as nn

class PointwiseRanker:
  def __init__(self, pointwise_scorer, documents):
    self.pointwise_scorer = pointwise_scorer
    self.documents = documents

  def __call__(self, query):
    ranks = []
    for query in query:
      scores = self.pointwise_scorer(torch.unsqueeze(query, 0),
                                     self.documents)
      _, sort_idx = torch.sort(scores, descending=True)
      ranks.append(sort_idx)
    return torch.stack(ranks)
