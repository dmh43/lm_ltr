import torch
import torch.nn as nn

from .pointwise_scorer import PointwiseScorer

class PairwiseScorer(nn.Module):
  def __init__(self,
               query_token_embeds,
               document_token_embeds,
               doc_encoder):
    super().__init__()
    self.pointwise_scorer = PointwiseScorer(query_token_embeds, document_token_embeds, doc_encoder)

  def forward(self, query, document_1, document_2):
    score_1 = self.pointwise_scorer(query, document_1)
    score_2 = self.pointwise_scorer(query, document_2)
    return score_1 - score_2
