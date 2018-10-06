import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

class DocumentEncoder(nn.Module):
  def __init__(self, document_token_embeds):
    super().__init__()
    self.document_token_embeds = document_token_embeds
    self.weights = nn.Embedding(len(document_token_embeds.weight), 1)
    self.lstm = nn.LSTM(input_size=document_token_embeds.weight.shape[1],
                        hidden_size=100)
    # MultiBatchRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
    #                              dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, qrnn=qrnn)

  def forward(self, packed_document_and_order):
    packed_document, order = packed_document_and_order
    # document_tokens = self.document_token_embeds(packed_document)
    # out, last_out_and_last_cell = self.lstm(document_tokens)
    # last_out, last_cell_state = last_out_and_last_cell
    # return last_out.squeeze()[order]
    document = pad_packed_sequence(torch.nn.utils.rnn.PackedSequence(packed_document),
                                   padding_value=1.0,
                                   batch_first=True)[0][order]
    document_tokens = self.document_token_embeds(document)
    token_weights = self.weights(document)
    normalized_weights = F.softmax(token_weights, 1)
    doc_vecs = torch.sum(normalized_weights * document_tokens, 1)
    return doc_vecs / (torch.norm(doc_vecs, 2, 1).unsqueeze(1) + 0.0001)
