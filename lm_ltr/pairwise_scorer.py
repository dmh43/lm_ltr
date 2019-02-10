import torch
import torch.nn as nn

from .pointwise_scorer import PointwiseScorer

class PairwiseScorer(nn.Module):
  def __init__(self,
               query_token_embeds,
               document_token_embeds,
               doc_encoder,
               model_params,
               train_params,
               use_bow_model=False):
    super().__init__()
    self.pointwise_scorer = PointwiseScorer(query_token_embeds,
                                            document_token_embeds,
                                            doc_encoder,
                                            model_params,
                                            train_params,
                                            use_bow_model=use_bow_model)

  def forward(self, query, document_1, document_2, lens_1, lens_2, doc_1_score, doc_2_score, to_idx=None):
    score_1 = self.pointwise_scorer(query, document_1, lens_1, doc_1_score, to_idx=to_idx)
    score_2 = self.pointwise_scorer(query, document_2, lens_2, doc_2_score, to_idx=to_idx)
    return score_1 - score_2
