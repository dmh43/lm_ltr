import pydash as _

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence

from .preprocessing import pack

class PointwiseRanker:
  def __init__(self, device, pointwise_scorer):
    self.device = device
    self.pointwise_scorer = pointwise_scorer

  def __call__(self, query, documents):
    assert len(query.shape) == 2, "PointwiseRanker expects a single batch of queries"
    ranks = []
    packed_doc_and_order = pack(documents, self.device)
    for query in query.to(self.device):
      scores = self.pointwise_scorer(torch.unsqueeze(query, 0).repeat(len(documents), 1),
                                     packed_doc_and_order)
      sorted_scores, sort_idx = torch.sort(scores, descending=True)
      ranks.append(sort_idx)
    return torch.stack(ranks)
