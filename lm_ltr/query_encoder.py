import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryEncoder(nn.Module):
  def __init__(self, query_token_embeds, use_max_pooling=False):
    super().__init__()
    self.query_token_embeds = query_token_embeds
    self.weights = nn.Embedding(len(query_token_embeds.weight), 1)
    torch.nn.init.constant_(self.weights.data, 1)
    self.use_max_pooling = use_max_pooling
    self.max_pool = nn.AdaptiveMaxPool1d(1)

  def forward(self, query):
    query_tokens = self.query_token_embeds(query)
    token_weights = self.weights(query)
    normalized_weights = F.softmax(token_weights, 1)
    if self.use_max_pooling:
      pool_input = torch.transpose(normalized_weights * query_tokens, 0, 1)
      query_vecs = torch.squeeze(self.max_pool(pool_input))
    else:
      query_vecs = torch.sum(normalized_weights * query_tokens, 1)
    encoded = query_vecs
    return encoded
