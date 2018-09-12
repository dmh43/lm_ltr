import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryEncoder(nn.Module):
  def __init__(self, query_token_embeds):
    super().__init__()
    self.query_token_embeds = query_token_embeds
    self.weights = nn.Embedding(len(query_token_embeds.weight), 1)

  def forward(self, query):
    query_tokens = self.query_token_embeds(query)
    token_weights = self.weights(query)
    normalized_weights = F.softmax(token_weights, 1)
    return torch.sum(normalized_weights * query_tokens, 1)
