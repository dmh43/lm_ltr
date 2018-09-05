import torch
import torch.nn as nn

class QueryEncoder(nn.Module):
  def __init__(self, query_token_embeds: nn.Embedding, query_embed_len: int) -> None:
    super().__init__()
    self.query_token_embeds = query_token_embeds
    self.query_embed_len = query_embed_len
    self.linear = nn.Linear(self.query_embed_len, self.query_embed_len, bias=False)

  def forward(self, query):
    query_tokens = self.query_token_embeds(query)
    return self.linear(torch.sum(query_tokens, 1))
