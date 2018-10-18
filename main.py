import pickle
import random
import time

import pydash as _
import torch
import torch.nn as nn
from fastai.data import DataBunch

from lm_ltr.embedding_loaders import get_glove_lookup, init_embedding, extend_token_lookup
from lm_ltr.fetchers import get_raw_documents, get_supervised_raw_data, get_weak_raw_data, read_or_cache, read_cache, get_robust_documents, get_robust_queries, get_robust_rels, read_query_result
from lm_ltr.pointwise_scorer import PointwiseScorer
from lm_ltr.pairwise_scorer import PairwiseScorer
from lm_ltr.preprocessing import preprocess_raw_data, preprocess_texts, all_ones, score, inv_log_rank, inv_rank, exp_score, collate_query_samples, collate_query_pairwise_samples
from lm_ltr.data_wrappers import build_query_dataloader, build_query_pairwise_dataloader, RankingDataset
from lm_ltr.train_model import train_model
from lm_ltr.pretrained import get_doc_encoder_and_embeddings

def prepare_data():
  print('Loading mappings')
  with open('./document_ids.pkl', 'rb') as fh:
    document_title_id_mapping = pickle.load(fh)
    id_document_title_mapping = {document_title_id: document_title for document_title, document_title_id in _.to_pairs(document_title_id_mapping)}
  with open('./query_ids.pkl', 'rb') as fh:
    query_id_mapping = pickle.load(fh)
    id_query_mapping = {query_id: query for query, query_id in _.to_pairs(query_id_mapping)}
  print('Loading raw documents')
  raw_documents = read_or_cache('./raw_documents.pkl',
                                lambda: get_raw_documents(id_document_title_mapping))
  documents, document_token_lookup = preprocess_texts(raw_documents)
  size_train_queries = 0.8
  train_query_ids = random.sample(list(id_query_mapping.keys()),
                                  int(size_train_queries * len(id_query_mapping)))
  print('Loading weak data (from Indri output)')
  weak_raw_data = read_or_cache('./weak_raw_data.pkl',
                                lambda: get_weak_raw_data(id_query_mapping, train_query_ids))
  train_data, query_token_lookup = preprocess_raw_data(weak_raw_data)
  test_queries = [id_query_mapping[id] for id in set(id_query_mapping.keys()) - set(train_query_ids)]
  print('Loading supervised data (from mention-entity pairs)')
  supervised_raw_data = read_or_cache('./supervised_raw_data.pkl',
                                      lambda: get_supervised_raw_data(document_title_id_mapping, test_queries))
  test_data, __ = preprocess_raw_data(supervised_raw_data,
                                      query_token_lookup=query_token_lookup)
  return documents, train_data, test_data, query_token_lookup, document_token_lookup

def create_id_lookup(names_or_titles):
  return dict(zip(names_or_titles,
                  range(len(names_or_titles))))

def prepare(lookup, title_to_id):
  id_to_title_lookup = _.invert(title_to_id)
  ids = range(len(id_to_title_lookup))
  contents = [lookup[id_to_title_lookup[id]] for id in ids]
  numericalized, token_lookup = preprocess_texts(contents)
  return numericalized, token_lookup

def process_rels(query_name_document_title_rels, document_title_to_id, query_name_to_id, queries):
  data = []
  for query_name, doc_titles in query_name_document_title_rels.items():
    if query_name not in query_name_to_id: continue
    query_id = query_name_to_id[query_name]
    query = queries[query_id]
    if query is None: continue
    data.extend([{'query': query,
                  'doc_id': document_title_to_id[title]} for title in doc_titles if title in document_title_to_id])
  return data

model_to_save = None
def main():
  global model_to_save
  use_pretrained_doc_encoder = False
  use_pairwise_loss = True
  query_token_embed_len = 100
  document_token_embed_len = 100
  document_lookup = read_cache('./doc_lookup.pkl', get_robust_documents)
  query_lookup = get_robust_queries()
  query_name_document_title_rels = get_robust_rels()
  query_name_to_id = create_id_lookup(query_lookup.keys())
  document_title_to_id = create_id_lookup(document_lookup.keys())
  documents, document_token_lookup = prepare(document_lookup, document_title_to_id)
  queries, query_token_lookup = prepare(query_lookup, query_name_to_id)
  test_data = process_rels(query_name_document_title_rels,
                           document_title_to_id,
                           query_name_to_id,
                           queries)
  train_data = read_query_result(query_name_to_id, document_title_to_id, queries)
  glove_lookup = get_glove_lookup()
  num_query_tokens = len(query_token_lookup)
  num_doc_tokens = len(document_token_lookup)
  query_token_embeds = init_embedding(glove_lookup,
                                      query_token_lookup,
                                      num_query_tokens,
                                      query_token_embed_len)
  document_token_embeds = init_embedding(glove_lookup,
                                         document_token_lookup,
                                         num_doc_tokens,
                                         document_token_embed_len)
  extend_token_lookup(glove_lookup.keys(), document_token_lookup)
  extend_token_lookup(glove_lookup.keys(), query_token_lookup)
  doc_encoder = None
  if use_pretrained_doc_encoder:
    doc_encoder, document_token_embeds = get_doc_encoder_and_embeddings(document_token_lookup)
    doc_encoder.requires_grad = False
  if use_pairwise_loss:
    train_dl = build_query_pairwise_dataloader(documents, train_data, rel_method=score)
    test_dl = build_query_pairwise_dataloader(documents, test_data, rel_method=score)
    model = PairwiseScorer(query_token_embeds, document_token_embeds, doc_encoder)
  else:
    train_dl = build_query_dataloader(documents, train_data, rel_method=score)
    test_dl = build_query_dataloader(documents, test_data, rel_method=score)
    model = PointwiseScorer(query_token_embeds, document_token_embeds, doc_encoder)
  train_ranking_dataset = RankingDataset(documents,
                                         train_dl.dataset.rankings)
  test_ranking_dataset = RankingDataset(documents,
                                        test_dl.dataset.rankings)
  model_data = DataBunch(train_dl,
                         test_dl,
                         collate_fn=collate_query_pairwise_samples if use_pairwise_loss else collate_query_samples,
                         device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
  model_to_save = model
  train_model(model, model_data, train_ranking_dataset, test_ranking_dataset, use_pairwise_loss)

if __name__ == "__main__":
  import ipdb
  import traceback
  import sys

  try:
    main()
  except: # pylint: disable=bare-except
    if model_to_save:
      torch.save(model_to_save.state_dict(), './model_save_debug' + str(time.time()))
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
  ipdb.post_mortem(tb)
