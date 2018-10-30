import torch
import torch.nn as nn

import lm_ltr.embedding_loaders as el

def test_from_doc_to_query_embeds():
  word_embed_len = 15
  num_doc_embeddings = 10
  document_token_embeds = nn.Embedding(num_doc_embeddings, word_embed_len)
  document_token_lookup = {'<unk>': 0, '<pad>': 1, 'hi': 2, 'go': 3, 'no': 4}
  query_token_lookup = {'<unk>': 0, '<pad>': 1, 'not': 2, 'hi': 3, 'no': 4}
  query_embeds = el.from_doc_to_query_embeds(document_token_embeds,
                                             document_token_lookup,
                                             query_token_lookup)
  embeds = torch.cat([document_token_embeds.weight.data[[0, 1]],
                      query_embeds.weight.data[[2]],
                      document_token_embeds.weight.data[[2, 4]]],
                     0)
  assert torch.equal(query_embeds.weight.data, embeds)
  assert not torch.equal(query_embeds.weight.data[3],
                         document_token_embeds.weight.data[3])
