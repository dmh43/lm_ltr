from fastai.lm_rnn import MultiBatchRNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from toolz import pipe

from query_encoder import QueryEncoder
from document_encoder import DocumentEncoder

class PointwiseScorer(nn.Module):
  def __init__(self,
               query_token_embeds,
               document_token_embeds):
    super().__init__()
    self.document_encoder = DocumentEncoder(document_token_embeds)
    self.query_encoder = QueryEncoder(query_token_embeds)
    concat_len = 200
    self.to_logits = nn.Linear(concat_len, 1)

  def forward(self, query, document):
    hidden = torch.cat([self.document_encoder(document),
                        self.query_encoder(query)],
                       1)
    return pipe(hidden,
                self.to_logits,
                torch.squeeze)
    # return torch.sum(self.document_encoder(document) * self.query_encoder(query),
    #                  dim=1)
