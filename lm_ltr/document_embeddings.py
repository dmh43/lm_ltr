from typing import List

import torch
import torch.nn as nn
from fastai import MultiBatchRNN

from encode_documents import DocumentEncoder

class DocumentEmbedding(nn.Embedding):
  def __init__(self, documents: List[str], embedding_dim=100):
    self.encode_documents = DocumentEncoder(embedding_dim)
    encoded_documents = self.encode_documents(documents)
    super().__init__(len(documents),
                     embedding_dim,
                     _weight=encoded_documents)
