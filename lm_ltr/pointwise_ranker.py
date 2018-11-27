import pydash as _

import torch
import torch.nn as nn

from .preprocessing import pad
from .utils import at_least_one_dim

class PointwiseRanker:
  def __init__(self, device, pointwise_scorer, doc_chunk_size=-1):
    self.device = device
    self.pointwise_scorer = pointwise_scorer
    self.doc_chunk_size = doc_chunk_size

  def _scores_for_chunk(self, query, documents) -> None:
    if isinstance(documents, torch.Tensor) and len(documents.shape) == 1:
      padded_doc = documents
      lens = torch.zeros_like(documents)
    else:
      padded_doc, lens = pad(documents, self.device)
    with torch.no_grad():
      try:
        self.pointwise_scorer.eval()
        scores = self.pointwise_scorer(torch.unsqueeze(query, 0).repeat(len(documents), 1),
                                       padded_doc,
                                       lens)
      finally:
        self.pointwise_scorer.train()
    return at_least_one_dim(scores)

  def __call__(self, query, documents, k=None):
    assert len(query.shape) == 2, "PointwiseRanker expects a single batch of queries"
    k = k if k is not None else len(documents)
    ranks = []
    for query in query.to(self.device):
      if self.doc_chunk_size != -1:
        all_scores = []
        for from_idx, to_idx in zip(range(0,
                                          len(documents),
                                          self.doc_chunk_size),
                                    range(self.doc_chunk_size,
                                          len(documents) + self.doc_chunk_size,
                                          self.doc_chunk_size)):
          all_scores.append(self._scores_for_chunk(query, documents[from_idx : to_idx]))
        scores = torch.cat(all_scores, 0)
      else:
        scores = self._scores_for_chunk(query, documents)
      topk_scores, topk_idxs = torch.topk(scores, k)
      sorted_scores, sort_idx = torch.sort(topk_scores, descending=True)
      ranks.append(topk_idxs[sort_idx])
    return torch.stack(ranks)
