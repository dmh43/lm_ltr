from toolz import pipe

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

class DocumentEncoder(nn.Module):
  def __init__(self, document_token_embeds, doc_encoder=None, use_cnn=False):
    super().__init__()
    self.document_token_embeds = document_token_embeds
    self.use_cnn = use_cnn
    word_embed_size = document_token_embeds.weight.shape[1]
    self.use_lm = False
    if doc_encoder:
      self.use_lm = True
      self.pretrained_enc = doc_encoder
    else:
      self.weights = nn.Embedding(len(document_token_embeds.weight), 1)
      if self.use_cnn:
        num_filters = 50
        self.cnn = nn.Conv1d(word_embed_size, num_filters, 5)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.projection = nn.Linear(num_filters, 100)
      else:
        hidden_size = 100
        self.lstm = nn.LSTM(input_size=word_embed_size,
                            hidden_size=hidden_size,
                            bidirectional=True,
                            batch_first=True)
        self.projection = nn.Linear(hidden_size * 2, 100)

  def _lm_forward(self, packed_document_and_order):
    packed_document, order = packed_document_and_order
    packed_document = torch.nn.utils.rnn.PackedSequence(packed_document[0],
                                                        packed_document[1].to(torch.device('cpu')))
    unsorted_padded_docs = pad_packed_sequence(packed_document,
                                               padding_value=1)[0][:, order]
    doc_vecs = self.pretrained_enc(unsorted_padded_docs)
    return doc_vecs

  def _lstm_forward(self, packed_document_and_order):
    packed_document, order = packed_document_and_order
    packed_document_tokens = self.document_token_embeds(packed_document[0])
    packed_seq_document_tokens = torch.nn.utils.rnn.PackedSequence(packed_document_tokens,
                                                                   packed_document[1].to(torch.device('cpu')))
    out, last_out_and_last_cell = self.lstm(packed_seq_document_tokens)
    last_out, last_cell_state = last_out_and_last_cell
    doc_vecs = self.projection(torch.cat([last_out[0], last_out[1]], 1)[order])
    return doc_vecs

  def _weighted_forward(self, packed_document_and_order):
    packed_document, order = packed_document_and_order
    packed_document = torch.nn.utils.rnn.PackedSequence(packed_document[0],
                                                        packed_document[1].to(torch.device('cpu')))
    document = pad_packed_sequence(packed_document,
                                   padding_value=1,
                                   batch_first=True)[0][order]
    document_tokens = self.document_token_embeds(document)
    token_weights = self.weights(document)
    normalized_weights = F.softmax(token_weights, 1)
    doc_vecs = torch.sum(normalized_weights * document_tokens, 1)
    encoded = doc_vecs
    return encoded

  def _cnn_forward(self, packed_document_and_order):
    packed_document, order = packed_document_and_order
    packed_document = torch.nn.utils.rnn.PackedSequence(packed_document[0],
                                                        packed_document[1].to(torch.device('cpu')))
    document = pad_packed_sequence(packed_document,
                                   padding_value=1,
                                   batch_first=True)[0][order]
    document_tokens = self.document_token_embeds(document)
    return pipe(document_tokens,
                lambda batch: torch.transpose(batch, 1, 2),
                self.cnn,
                self.relu,
                self.pool,
                torch.squeeze,
                self.projection)

  def forward(self, packed_document_and_order):
    if self.use_cnn:
      return self._cnn_forward(packed_document_and_order)
    elif self.use_lm:
      return self._lm_forward(packed_document_and_order)
    else:
      return self._weighted_forward(packed_document_and_order)
      # return self._lstm_forward(packed_document_and_order)
