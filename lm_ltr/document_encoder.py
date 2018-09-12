import torch
import torch.nn as nn
import torch.nn.functional as F

class DocumentEncoder(nn.Module):
  def __init__(self, document_token_embeds):
    super().__init__()
    self.document_token_embeds = document_token_embeds
    self.weights = nn.Embedding(len(document_token_embeds.weight), 1)
    # MultiBatchRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
    #                              dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, qrnn=qrnn)

  def forward(self, document):
    document_tokens = self.document_token_embeds(document)
    token_weights = self.weights(document)
    normalized_weights = F.softmax(token_weights, 1)
    return torch.sum(normalized_weights * document_tokens, 1)
