import pydash as _

import torch
import torch.nn as nn

from .preprocessing import pad, _collate_bow_doc
from .utils import at_least_one_dim

def smooth_scores(scores, doc_scores, smooth):
  normalized_doc_scores = doc_scores / doc_scores.sum()
  return scores * (1 - smooth) + normalized_doc_scores * smooth

class PointwiseRanker:
  def __init__(self, device, pointwise_scorer, doc_chunk_size=-1, use_doc_scores_for_smoothing=False):
    self.device = device
    self.pointwise_scorer = pointwise_scorer
    self.doc_chunk_size = doc_chunk_size
    self.use_doc_scores_for_smoothing = use_doc_scores_for_smoothing

  def _scores_for_chunk(self, query, documents, doc_scores) -> None:
    if isinstance(documents, torch.Tensor) and len(documents.shape) == 1:
      padded_doc = documents
      lens = torch.zeros_like(documents)
    elif isinstance(documents[0], torch.Tensor):
      padded_doc, lens = pad(documents, self.device)
    else:
      padded_doc = tuple([tens.to(self.device) for tens in _collate_bow_doc(documents)])
      lens = torch.tensor([sum(doc.values()) for doc in documents],
                          device=self.device)
    with torch.no_grad():
      try:
        self.pointwise_scorer.eval()
        scores = self.pointwise_scorer(torch.unsqueeze(query, 0).repeat(len(documents), 1),
                                       padded_doc,
                                       lens,
                                       doc_scores if not self.use_doc_scores_for_smoothing else torch.zeros_like(doc_scores))
      finally:
        self.pointwise_scorer.train()
    return at_least_one_dim(scores)

  def __call__(self, query, documents, doc_scores, k=None, smooth=None):
    assert len(query.shape) == 2, "PointwiseRanker expects a single batch of queries"
    k = k if k is not None else len(documents)
    ranks = []
    doc_scores = doc_scores.to(self.device)
    for query in query.to(self.device):
      if self.doc_chunk_size != -1:
        all_scores = []
        for from_idx, to_idx in zip(range(0,
                                          len(documents),
                                          self.doc_chunk_size),
                                    range(self.doc_chunk_size,
                                          len(documents) + self.doc_chunk_size,
                                          self.doc_chunk_size)):
          all_scores.append(self._scores_for_chunk(query,
                                                   documents[from_idx : to_idx],
                                                   doc_scores[from_idx : to_idx]))
        scores = torch.cat(all_scores, 0)
      else:
        scores = self._scores_for_chunk(query, documents, doc_scores)
      if self.use_doc_scores_for_smoothing:
        assert smooth is not None, 'must specify smoothing amount'
        scores = smooth_scores(scores, doc_scores, smooth)
      topk_scores, topk_idxs = torch.topk(scores, k)
      sorted_scores, sort_idx = torch.sort(topk_scores, descending=True)
      ranks.append(topk_idxs[sort_idx])
    return torch.stack(ranks)
