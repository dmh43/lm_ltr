from typing import List

import torch
import torch.nn as nn

class DocumentEncoder(nn.Module):
  def __init__(self, document_embed_len):
    super().__init__(self)
    self.document_embed_len = document_embed_len
    # MultiBatchRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
    #                              dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, qrnn=qrnn)

  def forward(self, document: List[List[int]]) -> torch.Tensor:
    return torch.zeros(len(document), self.document_embed_len)
