import torch
import torch.nn as nn

class PointwiseRanker:
  def __init__(self, pointwise_scorer, expect_all_documents: bool):
    self.pointwise_scorer = pointwise_scorer
    self.expect_all_documents = expect_all_documents
    if not self.expect_all_documents:
      raise NotImplementedError

  def __call__(self, query, documents):
    documents_to_score = documents
    ranks = []
    for query in query:
      scores = self.pointwise_scorer(torch.unsqueeze(query, 0), documents_to_score, 0)
      _, sort_idx = torch.sort(scores)
      ranks.append(sort_idx)
    return torch.stack(ranks)
