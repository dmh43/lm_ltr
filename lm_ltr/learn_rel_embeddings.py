from progressbar import progressbar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, BatchSampler, RandomSampler, DataLoader

from lm_ltr.embedding_loaders import get_glove_lookup, init_embedding, extend_token_lookup
from lm_ltr.fetchers import read_cache, get_wiki_pages, get_mention_page_title_pairs, get_wiki_mentions
from lm_ltr.preprocessing import prepare, create_id_lookup, pad_to_max_len
from lm_ltr.utils import dont_update

def main():
  document_token_embed_len = 100
  query_token_embed_len= 100
  num_neg_samples = 4
  document_lookup = read_cache('./wiki_page_lookup.pkl', get_wiki_pages)
  document_title_to_id = create_id_lookup(document_lookup.keys())
  documents, document_token_lookup = read_cache('./parsed_wiki_pages.pkl',
                                                lambda: prepare(document_lookup, document_title_to_id))
  query_lookup = read_cache('./wiki_mention_lookup.pkl', get_wiki_mentions)
  query_name_to_id = create_id_lookup(query_lookup.keys())
  queries, query_token_lookup = read_cache('./parsed_wiki_mentions.pkl',
                                           lambda: prepare(query_lookup, query_name_to_id))
  rel_query_document_pairs = read_cache('./mention_entity_pairs.pkl',
                                        lambda: [[query_name_to_id[pair[0]],
                                                  document_title_to_id[pair[1]]]
                                                 for pair in get_mention_page_title_pairs()
                                                 if pair[1] in document_title_to_id])
  dataset = TensorDataset(torch.tensor(rel_query_document_pairs))
  batch_sampler = BatchSampler(RandomSampler(dataset), 100, False)
  dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
  num_doc_tokens = len(document_token_lookup)
  glove_lookup = get_glove_lookup()
  document_token_embeds = init_embedding(glove_lookup,
                                         document_token_lookup,
                                         num_doc_tokens,
                                         document_token_embed_len)
  dont_update(document_token_embeds)
  num_query_tokens = len(query_token_lookup)
  query_token_embeds = init_embedding(glove_lookup,
                                      query_token_lookup,
                                      num_query_tokens,
                                      query_token_embed_len)
  doc_tensors = torch.tensor(pad_to_max_len([doc[:num_doc_tokens] for doc in documents]))
  query_tensors = torch.tensor(pad_to_max_len(queries))
  device = torch.device('cuda')
  query_token_embeds = query_token_embeds.to(device)
  document_token_embeds = document_token_embeds.to(device)
  opt = torch.optim.Adam(query_token_embeds.parameters())
  for batch in progressbar(dataloader, max_value=len(batch_sampler)):
    batch = batch[0]
    opt.zero_grad()
    negative_doc_ids = torch.randint(len(document_lookup), (len(batch), num_neg_samples), device=device)
    doc_batch = document_token_embeds(doc_tensors[torch.transpose(batch, 0, 1)[1]].to(device))
    query_batch = query_token_embeds(query_tensors[torch.transpose(batch, 0, 1)[0]].to(device))
    neg_docs = document_token_embeds(doc_tensors[negative_doc_ids].to(device))
    pos_scores = F.tanh(torch.sum(doc_batch.sum(1) * query_batch.sum(1), 1))
    neg_scores = torch.sum(F.tanh(torch.sum(neg_docs.sum(2) * query_batch.sum(1).unsqueeze(1), 2)), 1)
    margin_violation = 1 - pos_scores + neg_scores
    loss = torch.max(torch.zeros_like(margin_violation), margin_violation).sum() / len(batch)
    loss.backward()
    opt.step()
  torch.save(query_token_embeds.state_dict(), './rel_embeds_save')


if __name__ == "__main__":
  import ipdb
  import traceback
  import sys

  try:
    main()
  except: # pylint: disable=bare-except
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
  ipdb.post_mortem(tb)
