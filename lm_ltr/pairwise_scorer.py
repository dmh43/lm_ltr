import torch
import torch.nn as nn

from .pointwise_scorer import PointwiseScorer

class PairwiseScorer(nn.Module):
  def __init__(self,
               query_token_embeds,
               document_token_embeds,
               doc_encoder,
               model_params,
               train_params):
    super().__init__()
    self.pointwise_scorer = PointwiseScorer(query_token_embeds,
                                            document_token_embeds,
                                            doc_encoder,
                                            model_params,
                                            train_params)

  def forward(self, query, document_1, document_2, order_1, order_2):
    batch_range_1, unsort_order_1 = torch.sort(order_1)
    batch_range_2, unsort_order_2 = torch.sort(order_2)
    score_1 = self.pointwise_scorer(query[order_1], document_1)
    score_2 = self.pointwise_scorer(query[order_2], document_2)
    return score_1[unsort_order_1] - score_2[unsort_order_2]
