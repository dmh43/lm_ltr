import torch
import torch.nn as nn

from lm_ltr.document_encoder import DocumentEncoder

def test_forward():
  num_words = 100
  word_embed_dim = 100
  word_embeds = nn.Embedding(num_words, word_embed_dim)
  encoder = DocumentEncoder(word_embeds)
  document = torch.tensor([[0, 0, 1, 1, 0, 2]])
  embedded = encoder((nn.utils.rnn.pack_sequence(document),
                      torch.arange(len(document))))
  assert embedded.shape[1] == word_embed_dim
