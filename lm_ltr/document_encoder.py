from toolz import pipe

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence, pack_padded_sequence


def repackage_hidden(h):
  if isinstance(h, torch.Tensor):
    return h.detach()
  else:
    return tuple(repackage_hidden(v) for v in h)

class DocumentEncoder(nn.Module):
  def __init__(self,
               document_token_embeds,
               doc_encoder=None,
               use_cnn=False,
               use_lstm=False,
               lstm_hidden_size=None):
    super().__init__()
    self.document_token_embeds = document_token_embeds
    self.use_cnn = use_cnn
    self.use_lstm = use_lstm
    self.lstm_hidden_size = lstm_hidden_size
    word_embed_size = document_token_embeds.weight.shape[1]
    self.use_lm = False
    if doc_encoder:
      self.use_lm = True
      self.pretrained_enc = doc_encoder
    else:
      self.weights = nn.Embedding(len(document_token_embeds.weight), 1)
      if self.use_cnn:
        num_filters = int(document_token_embeds.weight.shape[1] / 2)
        self.cnn = nn.Conv1d(word_embed_size, num_filters, 5)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.projection = nn.Linear(num_filters, document_token_embeds.weight.shape[1])
      elif self.use_lstm:
        self.num_lstm_layers = 1
        self.lstm = nn.LSTM(input_size=word_embed_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.num_lstm_layers,
                            bidirectional=True,
                            batch_first=True)
        self.projection = nn.Linear(self._get_pooled_dim(), 100)

  def _init_state(self, batch_size):
    weight = next(self.lstm.parameters())
    return (weight.new_zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size),
            weight.new_zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size))

  def _lm_forward(self, document):
    seq_dim_first = torch.transpose(document, 0, 1)
    doc_vecs = self.pretrained_enc(seq_dim_first)
    return doc_vecs

  def _lstm_forward(self, document, lens):
    pooled = self._pool(self._tbptt(document, lens))
    doc_vecs = self.projection(pooled)
    return doc_vecs

  def _lstm(self, document, state, lens):
    document_tokens = self.document_token_embeds(document)
    packed_document_tokens = pack_padded_sequence(document_tokens, lens, batch_first=True)
    return self.lstm(packed_document_tokens, state)

  def _tbptt(self, document, lens):
    bs, sl = document.size()
    outputs = []
    state = self._init_state(bs)
    for i in range(0, sl, self.bptt):
      o, new_state = self._lstm(document[:, i: min(i+self.bptt, sl)],
                                state,
                                lens[:, i: min(i+self.bptt, sl)])
      state = repackage_hidden(new_state)
      if i>(sl-self.max_seq):
        outputs.append(o)
    return self.concat(outputs)

  def _weighted_forward(self, document):
    document_tokens = self.document_token_embeds(document)
    token_weights = self.weights(document)
    normalized_weights = F.softmax(token_weights, 1)
    doc_vecs = torch.sum(normalized_weights * document_tokens, 1)
    encoded = doc_vecs
    return encoded

  def _cnn_forward(self, document):
    document_tokens = self.document_token_embeds(document)
    return pipe(document_tokens,
                lambda batch: torch.transpose(batch, 1, 2),
                self.cnn,
                self.relu,
                self.pool,
                torch.squeeze,
                self.projection)

  def forward(self, document, lens):
    if self.use_cnn:
      return self._cnn_forward(document)
    elif self.use_lm:
      return self._lm_forward(document)
    elif self.use_lstm:
      return self._lstm_forward(document, lens)
    else:
      return self._weighted_forward(document)
