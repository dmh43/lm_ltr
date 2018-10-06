import pydash as _

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence

class PointwiseRanker:
  def __init__(self, device, pointwise_scorer):
    self.device = device
    self.pointwise_scorer = pointwise_scorer

  def __call__(self, query, documents):
    assert len(query.shape) == 2, "PointwiseRanker expects a single batch of queries"
    ranks = []
    doc_lengths = torch.tensor(_.map_(documents, len), dtype=torch.long, device=self.device)
    sorted_doc_lengths, doc_order = torch.sort(doc_lengths, descending=True)
    sorted_doc = _.map_(doc_order, lambda idx: torch.tensor(documents[idx],
                                                            dtype=torch.long,
                                                            device=self.device))
    packed_doc_and_order = (pack_sequence(sorted_doc), doc_order)
    for query in query.to(self.device):
      scores = self.pointwise_scorer(torch.unsqueeze(query, 0).repeat(len(documents), 1),
                                     packed_doc_and_order)
      sorted_scores, sort_idx = torch.sort(scores, descending=True)
      ranks.append(sort_idx)
    return torch.stack(ranks)
