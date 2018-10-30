import pickle
import random
import time

import pydash as _
import torch
import torch.nn as nn
from fastai.data import DataBunch

from lm_ltr.embedding_loaders import get_glove_lookup, init_embedding, extend_token_lookup
from lm_ltr.fetchers import get_raw_documents, get_supervised_raw_data, get_weak_raw_data, read_or_cache, read_cache, get_robust_documents, get_robust_train_queries, get_robust_test_queries, get_robust_rels, read_query_result, read_query_test_rankings
from lm_ltr.pointwise_scorer import PointwiseScorer
from lm_ltr.pairwise_scorer import PairwiseScorer
from lm_ltr.preprocessing import preprocess_raw_data, preprocess_texts, all_ones, score, inv_log_rank, inv_rank, exp_score, collate_query_samples, collate_query_pairwise_samples, prepare, create_id_lookup, normalize_scores_query_wise
from lm_ltr.data_wrappers import build_query_dataloader, build_query_pairwise_dataloader, RankingDataset
from lm_ltr.train_model import train_model
from lm_ltr.pretrained import get_doc_encoder_and_embeddings
from lm_ltr.utils import dont_update

from rabbit_ml.rabbit_ml import Rabbit
from rabbit_ml.rabbit_ml.experiment import Experiment

args =  [{'name': 'batch_size',
          'for': 'train_params',
          'type': int,
          'default': 512},
         {'name': 'num_doc_tokens_to_consider',
          'for': 'train_params',
          'type': int,
          'default': 100},
         {'name': 'train_dataset_size',
          'for': 'train_params',
          'type': lambda size: int(size) if size is not None else None,
          'default': None},
         {'name': 'num_neg_samples',
          'for': 'train_params',
          'type': int,
          'default': 0},
         {'name': 'dropout_keep_prob',
          'for': 'train_params',
          'type': float,
          'default': 0.8},
         {'name': 'num_epochs',
          'for': 'train_params',
          'type': int,
          'default': 1},
         {'name': 'dont_freeze_word_embeds',
          'for': 'train_params',
          'type': 'flag',
          'default': False},
         {'name': 'use_pretrained_doc_encoder',
          'for': 'model_params',
          'type': 'flag',
          'default': False},
         {'name': 'dont_freeze_pretrained_doc_encoder',
          'for': 'train_params',
          'type': 'flag',
          'default': False},
         {'name': 'rel_method',
          'for': 'train_params',
          'type': eval,
          'default': score},
         {'name': 'use_pointwise_loss',
          'for': 'train_params',
          'type': 'flag',
          'default': False},
         {'name': 'use_gradient_clipping',
          'for': 'train_params',
          'type': 'flag',
          'default': False},
         {'name': 'gradient_clipping_norm',
          'for': 'train_params',
          'type': float,
          'default': 0.1},
         {'name': 'weight_decay',
          'for': 'train_params',
          'type': float,
          'default': 0.0},
         {'name': 'use_cosine_similarity',
          'for': 'model_params',
          'type': 'flag',
          'default': False},
         {'name': 'hidden_layer_sizes',
          'for': 'model_params',
          'type': lambda string: [int(size) for size in string.split(',')],
          'default': [128, 64, 16]},
         {'name': 'ablation',
          'for': 'model_params',
          'type': lambda string: string.split(','),
          'default': []},
         {'name': 'query_token_embed_len',
          'for': 'model_params',
          'type': int,
          'default': 100},
         {'name': 'query_token_embedding_set',
          'for': 'model_params',
          'type': str,
          'default': 'glove'},
         {'name': 'document_token_embed_len',
          'for': 'model_params',
          'type': int,
          'default': 100},
         {'name': 'document_token_embedding_set',
          'for': 'model_params',
          'type': str,
          'default': 'glove'},
         {'name': 'comments',
          'for': 'run_params',
          'type': str,
          'default': ''},
         {'name': 'load_model',
          'for': 'run_params',
          'type': 'flag',
          'default': False}]

class MyRabbit(Rabbit):
  def run(self):
    pass

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
  rabbit = MyRabbit(args)
  experiment = Experiment(rabbit.train_params + rabbit.model_params + rabbit.run_params)
  use_pretrained_doc_encoder = rabbit.model_params.use_pretrained_doc_encoder
  use_pointwise_loss = rabbit.train_params.use_pointwise_loss
  query_token_embed_len = rabbit.model_params.query_token_embed_len
  document_token_embed_len = rabbit.model_params.document_token_embed_len
  document_lookup = read_cache('./doc_lookup.pkl', get_robust_documents)
  num_doc_tokens_to_consider = rabbit.train_params.num_doc_tokens_to_consider
  document_title_to_id = read_cache('./document_title_to_id.pkl',
                                    lambda: create_id_lookup(document_lookup.keys()))
  documents, document_token_lookup = read_cache(f'./parsed_docs_{num_doc_tokens_to_consider}_tokens.pkl',
                                                lambda: prepare(document_lookup,
                                                                document_title_to_id,
                                                                num_tokens=num_doc_tokens_to_consider))
  train_query_lookup = read_cache('./robust_train_queries.pkl', get_robust_train_queries)
  train_query_name_to_id = read_cache('./train_query_name_to_id.pkl',
                                      lambda: create_id_lookup(train_query_lookup.keys()))
  train_queries, query_token_lookup = read_cache('./parsed_robust_queries.pkl',
                                                 lambda: prepare(train_query_lookup, train_query_name_to_id))
  train_data = read_cache(f'./robust_train_query_results_{num_doc_tokens_to_consider}_tokens',
                          lambda: read_query_result(train_query_name_to_id,
                                                    document_title_to_id,
                                                    train_queries))
  glove_lookup = get_glove_lookup()
  # extend_token_lookup(glove_lookup.keys(), document_token_lookup)
  # extend_token_lookup(glove_lookup.keys(), query_token_lookup)
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
  if not rabbit.train_params.dont_freeze_word_embeds:
    dont_update(document_token_embeds)
    dont_update(query_token_embeds)
  test_query_lookup = read_cache('./robust_test_queries.pkl',
                                 get_robust_test_queries)
  test_query_name_document_title_rels = read_cache('./robust_rels.pkl',
                                                   get_robust_rels)
  test_query_name_to_id = read_cache('./test_query_name_to_id.pkl',
                                     lambda: create_id_lookup(test_query_lookup.keys()))
  test_queries, __ = read_cache('./parsed_test_robust_queries.pkl',
                                lambda: prepare(test_query_lookup,
                                                test_query_name_to_id,
                                                token_lookup=query_token_lookup))
  test_data = read_cache('./parsed_robust_rels.pkl',
                         lambda: process_rels(test_query_name_document_title_rels,
                                              document_title_to_id,
                                              test_query_name_to_id,
                                              test_queries))
  doc_encoder = None
  if use_pretrained_doc_encoder:
    doc_encoder, document_token_embeds = get_doc_encoder_and_embeddings(document_token_lookup)
    if not rabbit.train_params.dont_freeze_pretrained_doc_encoder:
      dont_update(doc_encoder)
  if use_pointwise_loss:
    normalized_train_data = read_cache('./normalized_train_query_data.pkl',
                                       lambda: normalize_scores_query_wise(train_data))
    train_dl = build_query_dataloader(documents,
                                      normalized_train_data[:rabbit.train_params.train_dataset_size],
                                      rabbit.train_params.batch_size,
                                      rel_method=rabbit.train_params.rel_method)
    test_dl = build_query_dataloader(documents,
                                     test_data,
                                     rabbit.train_params.batch_size,
                                     rel_method=rabbit.train_params.rel_method)
    model = PointwiseScorer(query_token_embeds,
                            document_token_embeds,
                            doc_encoder,
                            rabbit.model_params,
                            rabbit.train_params)
  else:
    train_dl = build_query_pairwise_dataloader(documents,
                                               train_data[:rabbit.train_params.train_dataset_size],
                                               rabbit.train_params.batch_size,
                                               rel_method=rabbit.train_params.rel_method,
                                               num_neg_samples=rabbit.train_params.num_neg_samples)
    test_dl = build_query_pairwise_dataloader(documents,
                                              test_data,
                                              rabbit.train_params.batch_size,
                                              rel_method=rabbit.train_params.rel_method,
                                              num_neg_samples=0)
    model = PairwiseScorer(query_token_embeds,
                           document_token_embeds,
                           doc_encoder,
                           rabbit.model_params,
                           rabbit.train_params)
  train_ranking_dataset = RankingDataset(documents,
                                         train_dl.dataset.rankings,
                                         num_doc_tokens=num_doc_tokens_to_consider)
  test_ranking_candidates = read_cache('./test_ranking_candidates.pkl',
                                       read_query_test_rankings)
  lookup_by_title = lambda title: document_title_to_id.get(title) or 0
  test_ranking_candidates = _.map_values(test_ranking_candidates,
                                         lambda candidate_names: _.map_(candidate_names,
                                                                        lookup_by_title))
  test_ranking_candidates = _.to_pairs(test_ranking_candidates)
  test_ranking_candidates = _.map_(test_ranking_candidates,
                                   lambda pair: [test_queries[test_query_name_to_id[pair[0]]],
                                                 pair[1]])
  test_ranking_dataset = RankingDataset(documents,
                                        test_ranking_candidates,
                                        test_dl.dataset.rankings,
                                        num_doc_tokens=num_doc_tokens_to_consider)
  model_data = DataBunch(train_dl,
                         test_dl,
                         collate_fn=collate_query_samples if use_pointwise_loss else collate_query_pairwise_samples,
                         device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
  model_to_save = model
  del document_lookup
  del query_token_lookup
  del document_token_lookup
  del test_query_lookup
  del train_query_lookup
  del train_queries
  del test_queries
  del glove_lookup
  train_model(model,
              model_data,
              train_ranking_dataset,
              test_ranking_dataset,
              rabbit.train_params,
              rabbit.model_params,
              experiment)

if __name__ == "__main__":
  import ipdb
  import traceback
  import sys

  try:
    main()
  except: # pylint: disable=bare-except
    if model_to_save and input("save?") == 'y':
      torch.save(model_to_save.state_dict(), './model_save_debug' + str(time.time()))
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
  ipdb.post_mortem(tb)
