from typing import List

from fastai.lm_rnn import MultiBatchRNN
import torch
import torch.nn as nn

from query_encoder import QueryEncoder
from document_encoder import DocumentEncoder

class LMScorer(nn.Module):
  def __init__(self,
               query_term_embeds,
               bptt=70,
               max_seq=140,
               n_tok=30000,
               emb_sz=400,
               n_hid=1150,
               n_layers=3,
               pad_token=1,
               layers=[400*3, 50, 2],
               drops=[0.4, 0.1],
               bidir=False,
               dropouth=0.3,
               dropouti=0.5,
               dropoute=0.1,
               wdrop=0.5,
               qrnn=False,
               query_embed_len=100,
               document_embed_len=100):
    super().__init__(self)
    self.document_encoder = DocumentEncoder(document_embed_len)
    self.query_encoder = QueryEncoder(query_term_embeds, query_embed_len)
    hidden_len = document_embed_len + query_embed_len
    self.to_logits = nn.Linear(hidden_len, 2)

  def forward(self,
              candidate_documents: List[List[int]],
              query: List[List[int]]) -> torch.Tensor:
    hidden = torch.cat([self.document_encoder(candidate_documents),
                        self.query_encoder(query)],
                       1)
    return self.to_logits(hidden)
