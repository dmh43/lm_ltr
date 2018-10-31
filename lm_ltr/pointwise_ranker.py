import pydash as _

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence

from .preprocessing import pack

class PointwiseRanker:
  def __init__(self, device, pointwise_scorer, doc_chunk_size=-1):
    self.device = device
    self.pointwise_scorer = pointwise_scorer
    self.doc_chunk_size = doc_chunk_size

  def _call_on_chunk(self, ranks, query, documents, k=None) -> None:
    packed_doc_and_order = pack(documents, self.device)
    for query in query.to(self.device):
      scores = self.pointwise_scorer(torch.unsqueeze(query, 0).repeat(len(documents), 1),
                                     packed_doc_and_order)
      topk_scores, topk_idxs = torch.topk(scores, k)
      sorted_scores, sort_idx = torch.sort(topk_scores, descending=True)
      ranks.append(topk_idxs[sort_idx])

  def __call__(self, query, documents, k=None):
    assert len(query.shape) == 2, "PointwiseRanker expects a single batch of queries"
    k = k if k is not None else len(documents)
    ranks = []
    if self.doc_chunk_size != -1:
      for from_idx, to_idx in zip(range(0,
                                      len(documents),
                                        self.doc_chunk_size),
                                  range(self.doc_chunk_size,
                                        len(documents) + self.doc_chunk_size,
                                        self.doc_chunk_size)):
        self._call_on_chunk(ranks, query, documents[from_idx : to_idx], k)
    else:
      self._call_on_chunk(ranks, query, documents, k)
    return torch.stack(ranks)
