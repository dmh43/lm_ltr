import pydash as _

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import get_term_matching_tensor

class TermMatchingScorer(nn.Module):
  def __init__(self, query_document_token_mapping):
    super().__init__()
    self.query_document_token_mapping = query_document_token_mapping
    self.num_document_tokens = max(query_document_token_mapping.values()) + 1
    self.weights = nn.Parameter(torch.randn(self.num_document_tokens))
    self.bias = nn.Parameter(torch.randn(1)[0])

  def get_counts_terms(self, query, document):
    device = query.device
    if len(query.shape) == 2 and query.shape[0] == 1:
      queries = query.repeat(document.shape[0], 1)
    elif len(query.shape) == 2:
      assert query.shape[0] == document.shape[0]
      queries = query
    else:
      raise ValueError('query has shape ', query.shape)
    counts = []
    terms = []
    for q_doc in zip(queries, document):
      qu, doc = q_doc
      q_counts, q_terms = get_term_matching_tensor(self.query_document_token_mapping, qu, doc)
      counts.append(q_counts)
      terms.append(q_terms)
    return torch.tensor(counts, device=device, dtype=torch.long), torch.tensor(terms, device=device, dtype=torch.long)

  def score(self, counts, terms):
    assert len(counts) == len(terms), "batch_size should be equal for counts and terms"
    assert (terms < self.num_document_tokens).all()
    batch_weights = self.weights[terms]
    # return F.sigmoid(torch.sum(counts.float() * batch_weights, 1) + self.bias)
    return torch.sum(counts.float() * batch_weights, 1) + self.bias

  def forward(self, query, document):
    counts, terms = self.get_counts_terms(query, document)
    return self.score(counts, terms)
