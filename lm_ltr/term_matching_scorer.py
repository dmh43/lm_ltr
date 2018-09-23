import torch
import torch.nn as nn
import torch.nn.functional as F

class TermMatchingScorer(nn.Module):
  def __init__(self, query_document_token_mapping):
    super().__init__()
    self.query_document_token_mapping = query_document_token_mapping
    self.num_document_tokens = len(set(query_document_token_mapping.values()))
    self.weights = nn.Parameter(torch.randn(self.num_document_tokens))
    self.bias = nn.Parameter(torch.randn(1)[0])

  def forward(self, counts, terms):
    assert len(counts) == len(terms), "batch_size should be equal for counts and terms"
    batch_weights = self.weights[terms]
    return F.sigmoid(torch.sum(counts.float() * batch_weights, 1) + self.bias)
