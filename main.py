import pickle
import random
import time

import pydash as _
import torch
import torch.nn as nn
from fastai.data import DataBunch

from lm_ltr.embedding_loaders import get_glove_lookup, init_embedding, extend_token_lookup
from lm_ltr.fetchers import get_raw_documents, get_supervised_raw_data, get_weak_raw_data, read_or_cache, read_cache, load_robust04_data_and_docs
from lm_ltr.pointwise_scorer import PointwiseScorer
from lm_ltr.pairwise_scorer import PairwiseScorer
from lm_ltr.preprocessing import preprocess_raw_data, preprocess_texts, all_ones, score, inv_log_rank, inv_rank, exp_score, collate_query_samples, collate_query_pairwise_samples
from lm_ltr.data_wrappers import build_query_dataloader, build_query_pairwise_dataloader, RankingDataset
from lm_ltr.train_model import train_model
from lm_ltr.pretrained import get_doc_encoder_and_embeddings

def get_model_and_dls(query_token_embed_len,
                      document_token_embed_len,
                      query_token_lookup,
                      document_token_lookup,
                      documents,
                      weak_data,
                      use_pairwise_loss,
                      use_pretrained_doc_encoder,
                      test_set,
                      train_set):
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
  if test_set == 'wiki' and train_set == 'wiki':
    train_data = weak_data[:int(len(weak_data) * 0.8)]
    test_data = weak_data[int(len(weak_data) * 0.8):]
    additional_docs = []
  elif test_set == 'robust04' and train_set == 'robust04':
    robust_data, robust_docs = load_robust04_data_and_docs(query_token_lookup,
                                                           document_token_lookup,
                                                           len(documents))
    additional_docs = robust_docs
    train_data = robust_data[:int(len(robust_data) * 0.8)]
    test_data = robust_data[int(len(robust_data) * 0.8):]
  elif test_set == 'robust04' and train_set == 'wiki':
    train_data = weak_data
    test_data, additional_docs = load_robust04_data_and_docs(query_token_lookup,
                                                             document_token_lookup,
                                                             len(documents))
  elif test_set == 'clueweb' or train_set == 'clueweb':
    raise NotImplementedError
  else:
    raise ValueError(test_set + ' is not a valid test set name')
  documents = documents + additional_docs
  doc_encoder = None
  if use_pretrained_doc_encoder:
    doc_encoder, document_token_embeds = get_doc_encoder_and_embeddings(document_token_lookup)
    doc_encoder.requires_grad = False
  if use_pairwise_loss:
    train_dl = build_query_pairwise_dataloader(documents, train_data, rel_method=score)
    test_dl = build_query_pairwise_dataloader(documents, test_data, rel_method=score)
    scorer = PairwiseScorer(query_token_embeds, document_token_embeds, doc_encoder)
  else:
    train_dl = build_query_dataloader(documents, train_data, rel_method=score)
    test_dl = build_query_dataloader(documents, test_data, rel_method=score)
    scorer = PointwiseScorer(query_token_embeds, document_token_embeds, doc_encoder)
  return scorer, train_dl, test_dl, additional_docs

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

model_to_save = None
def old_main():
  global model_to_save
  documents, weak_data, sup_data, query_token_lookup, document_token_lookup = read_cache('./prepared_data.pkl', prepare_data)
  query_token_embed_len = 100
  document_token_embed_len = 100
  use_pairwise_loss = True
  # use_pretrained_doc_encoder = True
  use_pretrained_doc_encoder = False
  test_set = 'robust04'
  # test_set = 'wiki'
  train_set = 'robust04'
  # train_set = 'wiki'
  model, train_dl, test_dl, additional_docs = get_model_and_dls(query_token_embed_len,
                                                          document_token_embed_len,
                                                          query_token_lookup,
                                                          document_token_lookup,
                                                          documents,
                                                          weak_data,
                                                          use_pairwise_loss,
                                                          use_pretrained_doc_encoder,
                                                          test_set,
                                                          train_set)
  query_document_token_mapping = {idx: document_token_lookup.get(token) or document_token_lookup['<unk>'] for token, idx in query_token_lookup.items()}
  print('Creating ranking datasets')
  num_doc_tokens = 100
  all_documents = documents + additional_docs
  train_ranking_dataset = RankingDataset(all_documents,
                                         train_dl.dataset.rankings,
                                         query_document_token_mapping,
                                         k=1 if train_set == 'wiki' else 10,
                                         num_doc_tokens=num_doc_tokens)
  test_ranking_dataset = RankingDataset(all_documents,
                                        test_dl.dataset.rankings,
                                        query_document_token_mapping,
                                        k=1 if test_set == 'wiki' else 10,
                                        num_doc_tokens=num_doc_tokens)
  model_data = DataBunch(train_dl,
                         test_dl,
                         collate_fn=collate_query_pairwise_samples if use_pairwise_loss else collate_query_samples,
                         device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
  model_to_save = model
  train_model(model, model_data, train_ranking_dataset, test_ranking_dataset, use_pairwise_loss)

def main():
  global model_to_save
  robust_data, robust_docs = load_robust04_data_and_docs(query_token_lookup,
                                                         document_token_lookup,
                                                         len(documents))
  query_token_embed_len = 100
  document_token_embed_len = 100
  use_pairwise_loss = True
  # use_pretrained_doc_encoder = True
  use_pretrained_doc_encoder = False
  model, train_dl, test_dl, additional_docs = get_model_and_dls(query_token_embed_len,
                                                          document_token_embed_len,
                                                          query_token_lookup,
                                                          document_token_lookup,
                                                          documents,
                                                          weak_data,
                                                          use_pairwise_loss,
                                                          use_pretrained_doc_encoder,
                                                          test_set,
                                                          train_set)
  query_document_token_mapping = {idx: document_token_lookup.get(token) or document_token_lookup['<unk>'] for token, idx in query_token_lookup.items()}
  print('Creating ranking datasets')
  num_doc_tokens = 100
  all_documents = documents + additional_docs
  train_ranking_dataset = RankingDataset(all_documents,
                                         train_dl.dataset.rankings,
                                         query_document_token_mapping,
                                         k=10,
                                         num_doc_tokens=num_doc_tokens)
  test_ranking_dataset = RankingDataset(all_documents,
                                        test_dl.dataset.rankings,
                                        query_document_token_mapping,
                                        k=10,
                                        num_doc_tokens=num_doc_tokens)
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
