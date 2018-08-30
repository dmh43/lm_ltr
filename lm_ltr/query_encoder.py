from typing import List

import torch
import torch.nn as nn

class QueryEncoder(nn.Module):
  def __init__(self, query_token_embeds: nn.Embedding, query_embed_len: int) -> None:
    super().__init__()
    self.query_token_embeds = query_token_embeds
    self.query_embed_len = query_embed_len

  def forward(self, query: List[List[int]]) -> torch.Tensor:
    query_tokens = self.query_token_embeds(query)
    return torch.sum(query_tokens, 1) / len(query_tokens)
