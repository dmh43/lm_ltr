import torch
import math

def get_glove_lookup(path='./glove/glove.6B.100d.txt', embedding_dim=100):
  lookup = {'<pad>': torch.zeros(size=(embedding_dim,), dtype=torch.float32),
            '<unk>': torch.randn(size=(embedding_dim,), dtype=torch.float32)}
  with open(path) as f:
    while True:
      line = f.readline()
      if line and len(line) > 0:
        split_line = line.strip().split(' ')
        lookup[split_line[0]] = torch.tensor([float(val) for val in split_line[1:]],
                                             dtype=torch.float32)
      else:
        break
  return lookup

def init_embedding(glove_lookup, token_index_lookup, num_tokens, embed_len):
  token_embed_weights = nn.Parameter(torch.Tensor(num_tokens,
                                                  embed_len))
  token_embed_weights.data.normal_(0, 1.0/math.sqrt(embed_len))
  for token, index in token_index_lookup.items():
    if token in glove_lookup:
      token_embed_weights.data[index] = glove_lookup[token]
  embedding = nn.Embedding(num_tokens, embed_len)
  embedding.weight = token_embed_weights
  return embedding
