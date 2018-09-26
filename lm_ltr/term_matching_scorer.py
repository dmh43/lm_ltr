import torch
import torch.nn as nn
import torch.nn.functional as F

class TermMatchingScorer(nn.Module):
  def __init__(self, query_document_token_mapping):
    super().__init__()
    self.query_document_token_mapping = query_document_token_mapping
    self.num_document_tokens = max(query_document_token_mapping.values()) + 1
    self.weights = nn.Parameter(torch.randn(self.num_document_tokens))
    self.bias = nn.Parameter(torch.randn(1)[0])

  def forward(self, counts, terms):
    assert len(counts) == len(terms), "batch_size should be equal for counts and terms"
    assert (terms < self.num_document_tokens).all()
    batch_weights = self.weights[terms]
    # return F.sigmoid(torch.sum(counts.float() * batch_weights, 1) + self.bias)
    return torch.sum(counts.float() * batch_weights, 1) + self.bias
