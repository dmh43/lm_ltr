from typing import List

import torch
import torch.nn as nn

class QueryEncoder(nn.Module):
  def __init__(self, query_term_embeds: nn.Embedding, query_embed_len: int) -> None:
    self.query_term_embeds = query_term_embeds
    self.query_embed_len = query_embed_len

  def forward(self, query: List[List[int]]) -> torch.Tensor:
    query_terms = self.query_term_embeds(torch.tensor(query))
    return torch.sum(query_terms, 1)
