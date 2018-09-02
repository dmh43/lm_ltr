from typing import List

import torch
import torch.nn as nn

class DocumentEncoder(nn.Module):
  def __init__(self,
               document_token_embeds: nn.Embedding,
               document_embed_len: int):
    super().__init__()
    self.document_token_embeds = document_token_embeds
    self.document_embed_len = document_embed_len
    self.linear = nn.Linear(self.document_embed_len, self.document_embed_len, bias=False)
    # MultiBatchRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
    #                              dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, qrnn=qrnn)

  def forward(self, document: List[List[int]]) -> torch.Tensor:
    document_tokens = self.document_token_embeds(document)
    return self.linear(torch.sum(document_tokens[:, :20], 1))
