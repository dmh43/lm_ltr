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

  def forward(self, query, packed_document_and_order):
    packed_document, order = packed_document_and_order
    packed_document = torch.nn.utils.rnn.PackedSequence(packed_document[0],
                                                        packed_document[1].to(torch.device('cpu')))
    document = pad_packed_sequence(packed_document,
                                   padding_value=1,
                                   batch_first=True)[0][order]
    document_tokens = self.document_token_embeds(document)[:, :self.num_pos_tokens]
    query_tokens = self.query_token_embeds(query)
    query_embeds = query_tokens.sum(1).unsqueeze(1)
    neg_doc_tokens = sample(range(self.nce_sample_mul * len(self.document_token_embeds.weight)),
                            self.num_pos_tokens)
    pos_posterior = F.sigmoid(- torch.sum(query_embeds * document_tokens, 2))
    neg_posterior = F.sigmoid(- torch.sum(query_embeds * neg_doc_tokens, 1))
    return torch.sum(torch.log(pos_posterior) + torch.log(neg_posterior))
