import torch
import torch.nn as nn

from lm_ltr.query_encoder import QueryEncoder

def test_forward():
  num_words = 100
  word_embed_dim = 100
  word_embeds = nn.Embedding(num_words, word_embed_dim)
  encoder = QueryEncoder(word_embeds)
  query = torch.tensor([[0, 0, 1, 1, 0, 2]])
  embedded = encoder(query)
  assert embedded.shape[1] == word_embed_dim
