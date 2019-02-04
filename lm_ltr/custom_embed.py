import torch
import torch.nn as nn

class CustomEmbedding(nn.Module):
  def __init__(self, inp, out, padding_idx=None):
    super().__init__()
    self.inp = inp
    self.out = out
    self.weight = nn.Parameter(torch.randn(inp, out))
    self.padding_idx = padding_idx
    if self.padding_idx is not None:
      self.weight.data[self.padding_idx].fill_(0)

  def forward(self, ind):
    ind_shape = ind.size()
    ind = ind.view(-1)
    emb = self.weight[ind]
    emb = emb.view(*ind_shape, self.out)
    return emb
