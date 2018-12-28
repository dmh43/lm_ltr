import torch
import torch.nn as nn
import torch.nn.functional as F
from toolz import pipe

from .query_encoder import QueryEncoder
from .document_encoder import DocumentEncoder
from .utils import Identity

def _get_layer(from_size, to_size, dropout_keep_prob, activation=None, use_layer_norm=False, use_batch_norm=False):
  return [nn.Linear(from_size, to_size),
          nn.LayerNorm(to_size) if use_layer_norm else Identity(),
          nn.BatchNorm1d(to_size) if use_batch_norm else Identity(),
          nn.ReLU() if activation is None else activation,
          nn.Dropout(1 - dropout_keep_prob)]

class PointwiseScorer(nn.Module):
  def __init__(self,
               query_token_embeds,
               document_token_embeds,
               doc_encoder,
               model_params,
               train_params):
    super().__init__()
    self.use_layer_norm = train_params.use_layer_norm
    self.use_batch_norm = train_params.use_batch_norm
    self.frame_as_qa = model_params.frame_as_qa
    self.document_encoder = DocumentEncoder(document_token_embeds,
                                            doc_encoder,
                                            model_params.use_cnn,
                                            model_params.use_lstm,
                                            model_params.lstm_hidden_size,
                                            model_params.use_doc_out,
                                            model_params.only_use_last_out,
                                            train_params.word_level_do_kp)
    self.query_encoder = QueryEncoder(query_token_embeds, model_params.use_max_pooling)
    if model_params.use_pretrained_doc_encoder or model_params.use_doc_out:
      if model_params.only_use_last_out:
        if model_params.use_glove:
          concat_len = 400 + model_params.query_token_embed_len
        else:
          concat_len = 800
      else:
        if model_params.use_glove:
          concat_len = 1200 + model_params.query_token_embed_len
        else:
          concat_len = 1600
    else:
      concat_len = model_params.document_token_embed_len + model_params.query_token_embed_len
    concat_len += 1
    self.layers = nn.ModuleList()
    if not model_params.use_cosine_similarity:
      from_size = concat_len + sum([model_params.query_token_embed_len
                                    for i in [model_params.append_hadamard, model_params.append_difference]
                                    if i])
      for to_size in model_params.hidden_layer_sizes:
        self.layers.extend(_get_layer(from_size,
                                      to_size,
                                      train_params.dropout_keep_prob,
                                      use_layer_norm=self.use_layer_norm,
                                      use_batch_norm=self.use_batch_norm))
        from_size = to_size
      self.layers.extend(_get_layer(from_size, 1, train_params.dropout_keep_prob, activation=Identity()))
    if not train_params.use_pointwise_loss:
      if not train_params.use_bce_loss:
        self.layers.append(nn.Tanh())
    self.use_cosine_similarity = model_params.use_cosine_similarity
    self.append_difference = model_params.append_difference
    self.append_hadamard = model_params.append_hadamard


  def forward(self, query, document, lens, doc_score):
    sorted_lens, sort_order = torch.sort(lens, descending=True)
    batch_range, unsort_order = torch.sort(sort_order)
    if self.frame_as_qa:
      qa = torch.cat([document[sort_order], query[sort_order]], 1)
      doc_embed = self.document_encoder(qa, sorted_lens)
      query_embed = torch.tensor([], device=doc_embed.device)
    else:
      doc_embed = self.document_encoder(document[sort_order], sorted_lens)
      query_embed = self.query_encoder(query)[sort_order]
    if self.use_cosine_similarity:
      hidden = torch.sum(doc_embed * query_embed, 1)
    else:
      hidden = torch.cat([doc_embed, query_embed], 1)
    if self.append_difference:
      hidden = torch.cat([hidden, doc_embed - query_embed], 1)
    if self.append_hadamard:
      hidden = torch.cat([hidden, doc_embed * query_embed], 1)
    if not self.use_cosine_similarity:
      hidden = torch.cat([hidden, doc_score.unsqueeze(1)], 1)
    return pipe(hidden,
                *self.layers,
                torch.squeeze)[unsort_order]
