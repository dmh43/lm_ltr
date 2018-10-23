import torch
import torch.nn as nn
import torch.nn.functional as F
from toolz import pipe

from .query_encoder import QueryEncoder
from .document_encoder import DocumentEncoder
from .utils import Identity

def _get_layer(from_size, to_size, dropout_keep_prob, activation=None):
  return [nn.Linear(from_size, to_size),
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
    self.document_encoder = DocumentEncoder(document_token_embeds, doc_encoder)
    self.query_encoder = QueryEncoder(query_token_embeds)
    if model_params.use_pretrained_doc_encoder:
      concat_len = 1300
    else:
      concat_len = model_params.document_token_embed_len + model_params.query_token_embed_len
    if model_params.use_deep_network:
      from_size = concat_len
      self.layers = nn.ModuleList()
      for to_size in model_params.hidden_layer_sizes:
        self.layers.extend(_get_layer(from_size, to_size, train_params.dropout_keep_prob))
        from_size = to_size
      self.layers.extend(_get_layer(from_size, 1, train_params.dropout_keep_prob, activation=Identity()))
    else:
      self.to_logits = nn.Linear(concat_len, 1)
    self.layers.append(nn.Tanh())
    self.use_deep_network = model_params.use_deep_network


  def forward(self, query, document):
    if self.use_deep_network:
      hidden = torch.cat([self.document_encoder(document),
                          self.query_encoder(query)],
                         1)
      return pipe(hidden,
                  *self.layers,
                  torch.squeeze)
    else:
      return self.tanh(torch.sum(self.document_encoder(document) * self.query_encoder(query), 1))
