import gensim
import torch
import torch.nn as nn
import math

def get_glove_lookup(path=None, embedding_dim=100, use_large_embed=False, use_word2vec=False):
  if use_word2vec:
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    assert embedding_dim == 300, 'only d=300 supported for word2vec'
    lookup = {'<pad>': torch.zeros(size=(300,), dtype=torch.float32),
              '<unk>': torch.randn(size=(300,), dtype=torch.float32)}
    word2vec = dict(zip(model.index2word,
                        (torch.tensor(vec) for vec in model.vectors)))
    word2vec['<pad>'] = torch.zeros(size=(300,), dtype=torch.float32)
    word2vec['<unk>'] = torch.randn(size=(300,), dtype=torch.float32)
    return word2vec
  else:
    if path is None:
      if use_large_embed:
        path = f'./glove/glove.840B.{embedding_dim}d.txt'
      else:
        path = f'./glove/glove.6B.{embedding_dim}d.txt'
    lookup = {'<pad>': torch.zeros(size=(embedding_dim,), dtype=torch.float32),
              '<unk>': torch.randn(size=(embedding_dim,), dtype=torch.float32)}
    try:
      f = open(path)
    except:
      f = open('./glove/glove.6B.100d.txt')
    for line in f:
      split_line = line.rstrip().split(' ')
      lookup[split_line[0]] = torch.tensor([float(val) for val in split_line[1:]],
                                           dtype=torch.float32)
    f.close()
    return lookup

def init_embedding(glove_lookup, token_lookup, num_tokens, embed_len):
  embedding = nn.Embedding(num_tokens, embed_len, padding_idx=1)
  token_embed_weights = nn.Parameter(torch.Tensor(num_tokens,
                                                    embed_len))
  token_embed_weights.data.normal_(0, 1.0/math.sqrt(embed_len))
  if embed_len in [100, 300]:
    for token, index in token_lookup.items():
      if token in glove_lookup:
        token_embed_weights.data[index] = glove_lookup[token]
  embedding.weight = token_embed_weights
  return embedding

def extend_token_lookup(tokens_to_insert, token_lookup) -> None:
  for token in tokens_to_insert:
    if token not in token_lookup:
      token_lookup[token] = len(token_lookup)

def from_doc_to_query_embeds(document_token_embeds,
                             document_token_lookup,
                             query_token_lookup):
  from_idxs = []
  to_idxs = []
  doc_weights = document_token_embeds.weight
  num_tokens = len(query_token_lookup)
  embed_len = len(doc_weights[0])
  weights = torch.Tensor(num_tokens, embed_len)
  weights.data.normal_(0, 1.0/math.sqrt(embed_len))
  for query_token, query_idx in query_token_lookup.items():
    if query_token not in document_token_lookup: continue
    doc_idx = document_token_lookup[query_token]
    from_idxs.append(doc_idx)
    to_idxs.append(query_idx)
  weights[to_idxs] = doc_weights[from_idxs]
  embedding = nn.Embedding(num_tokens, embed_len, padding_idx=1)
  embedding.weight = nn.Parameter(weights)
  return embedding

class AdditiveEmbedding(nn.Embedding):
  def __init__(self, embeds_init, num_tokens, embed_len, *args, **kwargs):
    super().__init__(num_tokens, embed_len, *args, **kwargs)
    self.embeds_init = embeds_init
    self.embedding = nn.Embedding(num_tokens, embed_len, padding_idx=1)
    additive_weights = nn.Parameter(torch.Tensor(num_tokens, embed_len))
    additive_weights.data.normal_(0, 1.0/math.sqrt(embed_len))
    self.embedding.weight = additive_weights

  def forward(self, idxs):
    return self.embeds_init(idxs) + self.embedding(idxs)

def get_additive_regularized_embeds(embeds_init):
  num_tokens = len(embeds_init.weight)
  embed_len = len(embeds_init.weight[0])
  embedding = AdditiveEmbedding(embeds_init, num_tokens, embed_len)
  return embedding, embedding.embedding
