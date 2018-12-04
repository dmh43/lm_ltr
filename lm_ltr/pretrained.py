import pickle

import pydash as _

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.text import convert_weights, get_language_model, SequentialRNN
from fastai.layers import bn_drop_lin

class OutPooler(nn.Module):
  def __init__(self, only_use_last_out):
    super().__init__()
    self.only_use_last_out = only_use_last_out

  def pool(self, x, bs:int, is_max:bool):
    "Pool the tensor along the seq_len dimension."
    f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
    return f(x.permute(1,2,0), (1,)).view(bs,-1)

  def forward(self, input):
    raw_outputs, outputs = input
    output = outputs[-1]
    if self.only_use_last_out:
      return output[-1]
    else:
      sl,bs,__ = output.size()
      avgpool = self.pool(output, bs, False)
      mxpool = self.pool(output, bs, True)
      return torch.cat([output[-1], mxpool, avgpool], 1)

def get_doc_encoder_and_embeddings(document_token_lookup, only_use_last_out=False):
  emb_sz = 400
  n_hid = 1150
  n_layers = 3
  pad_token = 1
  model = get_language_model(len(document_token_lookup), emb_sz, n_hid, n_layers, pad_token)
  wgts = torch.load('lstm_wt103.pth', map_location=lambda storage, loc: storage)
  with open('./itos_wt103.pkl', 'rb') as fh:
    old_itos = pickle.load(fh)
  old_stoi = _.invert(old_itos)
  string_lookup = _.invert(document_token_lookup)
  wgts = convert_weights(wgts,
                         old_stoi,
                         [string_lookup[i]
                          for i in range(len(document_token_lookup))])
  model.load_state_dict(wgts)
  rnn_enc = model[0]
  embedding = rnn_enc.encoder
  return SequentialRNN(rnn_enc, OutPooler(only_use_last_out)), embedding
