import torch
import torch.nn as nn

class DocumentEncoder(nn.Module):
  def __init__(self, embedding_dim):
