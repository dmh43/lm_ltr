from typing import List

import torch
import torch.nn as nn

from encode_query import QueryEncoder
from document_embedding import DocumentEmbedding

class InformationRetrieval(nn.Module):
  def __init__(self, texts: List[str]):
    super().__init__()
    self.query_encoder = QueryEncoder()
    self.encoded_documents = DocumentEmbedding(texts)

  def forward(self, query_text: str) -> torch.Tensor:
    encoded_query = self.query_encoder(query_text)
