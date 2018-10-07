import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

class DocumentEncoder(nn.Module):
  def __init__(self, document_token_embeds):
    super().__init__()
    self.document_token_embeds = document_token_embeds
    self.weights = nn.Embedding(len(document_token_embeds.weight), 1)
    hidden_size = 100
    self.lstm = nn.LSTM(input_size=document_token_embeds.weight.shape[1],
                        hidden_size=hidden_size,
                        bidirectional=True,
                        batch_first=True)
    self.projection = nn.Linear(hidden_size * 2, 100)
    # MultiBatchRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
    #                              dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, qrnn=qrnn)

  def _lstm_forward(self, packed_document_and_order):
    packed_document, order = packed_document_and_order
    packed_document_tokens = self.document_token_embeds(packed_document[0])
    packed_seq_document_tokens = torch.nn.utils.rnn.PackedSequence(packed_document_tokens,
                                                                   packed_document[1])
    out, last_out_and_last_cell = self.lstm(packed_seq_document_tokens)
    last_out, last_cell_state = last_out_and_last_cell
    batch_size = last_out.shape[1]
    hidden_size = last_out.shape[2]
    doc_vecs = self.projection(torch.cat([last_out[0], last_out[1]], 1)[order])
    return doc_vecs / (torch.norm(doc_vecs, 2, 1).unsqueeze(1) + 0.0001)

  def _weighted_forward(self, packed_document_and_order):
    packed_document, order = packed_document_and_order
    document = pad_packed_sequence(torch.nn.utils.rnn.PackedSequence(packed_document),
                                   padding_value=1,
                                   batch_first=True)[0][order]
    document_tokens = self.document_token_embeds(document)
    token_weights = self.weights(document)
    normalized_weights = F.softmax(token_weights, 1)
    doc_vecs = torch.sum(normalized_weights * document_tokens, 1)
    return doc_vecs / (torch.norm(doc_vecs, 2, 1).unsqueeze(1) + 0.0001)

  def forward(self, packed_document_and_order):
    # return self._weighted_forward(packed_document_and_order)
    return self._lstm_forward(packed_document_and_order)
