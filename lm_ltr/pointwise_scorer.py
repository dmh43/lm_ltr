import torch
import torch.nn as nn
import torch.nn.functional as F
from toolz import pipe

from .query_encoder import QueryEncoder
from .document_encoder import DocumentEncoder

class PointwiseScorer(nn.Module):
  def __init__(self,
               query_token_embeds,
               document_token_embeds,
               doc_encoder):
    super().__init__()
    self.document_encoder = DocumentEncoder(document_token_embeds, doc_encoder)
    self.query_encoder = QueryEncoder(query_token_embeds)
    if doc_encoder:
      concat_len = 1300
    else:
      concat_len = 200
    self.to_logits = nn.Linear(concat_len, 1)
    self.lin1 = nn.Linear(concat_len, int(concat_len/2))
    self.relu1 = nn.ReLU()
    # self.dropout = nn.Dropout(0.5)
    self.lin2 = nn.Linear(int(concat_len/2), 1)
    self.tanh = nn.Tanh()


  def forward(self, query, document):
    return self.tanh(torch.sum(self.document_encoder(document) * self.query_encoder(query), 1))
    hidden = torch.cat([self.document_encoder(document),
                        self.query_encoder(query)],
                       1)
    return pipe(hidden,
                self.lin1,
                self.relu1,
                # self.dropout,
                self.lin2,
                self.tanh,
                torch.squeeze)
    # return torch.sum(self.document_encoder(document) * self.query_encoder(query),
    #                  dim=1)
