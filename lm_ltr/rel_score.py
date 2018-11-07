from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from toolz import pipe

class RelScore(nn.Module):
  def __init__(self,
               query_token_embeds,
               document_token_embeds,
               model_params,
               train_params):
    super().__init__()
    self.query_token_embeds = query_token_embeds
    self.document_token_embeds = document_token_embeds
    self.num_pos_tokens = train_params.num_pos_tokens_rel_score
    self.nce_sample_mul = train_params.nce_sample_mul_rel_score

  def forward(self, query, document):
    batch_size = len(query)
    document_tokens = self.document_token_embeds(document)[:, :self.num_pos_tokens]
    query_tokens = self.query_token_embeds(query)
    query_embeds = query_tokens.sum(1).unsqueeze(1)
    neg_doc_tokens_idxs = sample(range(len(self.document_token_embeds.weight)),
                                 self.nce_sample_mul * self.num_pos_tokens)
    neg_doc_tokens = self.document_token_embeds(torch.tensor(neg_doc_tokens_idxs, device=query.device))
    pos_posterior = F.sigmoid(torch.sum(query_embeds * document_tokens, 2) / len(document_tokens))
    neg_posterior = F.sigmoid(torch.sum(query_embeds * neg_doc_tokens, 1) / len(neg_doc_tokens))
    return -(torch.sum(torch.log(pos_posterior)) + torch.sum(torch.log(neg_posterior))) / batch_size
