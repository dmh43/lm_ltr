import pickle
import json
import random
import time

import pydash as _
import torch
import torch.nn as nn
from fastai.basic_data import DataBunch

from lm_ltr.embedding_loaders import get_glove_lookup, init_embedding, extend_token_lookup, from_doc_to_query_embeds, get_additive_regularized_embeds
from lm_ltr.fetchers import get_raw_documents, get_supervised_raw_data, get_weak_raw_data, read_or_cache, read_cache, get_robust_documents, get_robust_train_queries, get_robust_test_queries, get_robust_rels, read_query_result, read_query_test_rankings, read_from_file
from lm_ltr.pointwise_scorer import PointwiseScorer
from lm_ltr.pairwise_scorer import PairwiseScorer
from lm_ltr.preprocessing import preprocess_texts, all_ones, score, inv_log_rank, inv_rank, exp_score, collate_query_samples, collate_query_pairwise_samples, prepare, create_id_lookup, normalize_scores_query_wise, process_rels
from lm_ltr.data_wrappers import build_query_dataloader, build_query_pairwise_dataloader, RankingDataset
from lm_ltr.train_model import train_model
from lm_ltr.pretrained import get_doc_encoder_and_embeddings
from lm_ltr.utils import dont_update, do_update
from lm_ltr.multi_objective import MultiObjective
from lm_ltr.rel_score import RelScore
from lm_ltr.regularization import Regularization

from rabbit_ml.rabbit_ml import Rabbit
from rabbit_ml.rabbit_ml.experiment import Experiment

args =  [{'name': 'ablation', 'for': 'model_params', 'type': lambda string: string.split(','), 'default': []},
         {'name': 'add_rel_score', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'batch_size', 'for': 'train_params', 'type': int, 'default': 512},
         {'name': 'comments', 'for': 'run_params', 'type': str, 'default': ''},
         {'name': 'document_token_embed_len', 'for': 'model_params', 'type': int, 'default': 100},
         {'name': 'document_token_embedding_set', 'for': 'model_params', 'type': str, 'default': 'glove'},
         {'name': 'dont_freeze_pretrained_doc_encoder', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'dont_freeze_word_embeds', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'dropout_keep_prob', 'for': 'train_params', 'type': float, 'default': 0.8},
         {'name': 'frame_as_qa', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'gradient_clipping_norm', 'for': 'train_params', 'type': float, 'default': 0.1},
         {'name': 'hidden_layer_sizes', 'for': 'model_params', 'type': lambda string: [int(size) for size in string.split(',')], 'default': [128, 64, 16]},
         {'name': 'just_caches', 'for': 'run_params', 'type': 'flag', 'default': False},
         {'name': 'learning_rate', 'for': 'train_params', 'type': float, 'default': 1e-3},
         {'name': 'dont_limit_num_uniq_tokens', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'load_model', 'for': 'run_params', 'type': 'flag', 'default': False},
         {'name': 'lstm_hidden_size', 'for': 'model_params', 'type': int, 'default': 100},
         {'name': 'margin', 'for': 'train_params', 'type': float, 'default': 1.0},
         {'name': 'memorize_test', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'nce_sample_mul_rel_score', 'for': 'train_params', 'type': int, 'default': 5},
         {'name': 'num_doc_tokens_to_consider', 'for': 'train_params', 'type': int, 'default': 100},
         {'name': 'num_epochs', 'for': 'train_params', 'type': int, 'default': 1},
         {'name': 'num_neg_samples', 'for': 'train_params', 'type': int, 'default': 0},
         {'name': 'num_pos_tokens_rel_score', 'for': 'train_params', 'type': int, 'default': 20},
         {'name': 'only_use_last_out', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'query_token_embed_len', 'for': 'model_params', 'type': int, 'default': 100},
         {'name': 'query_token_embedding_set', 'for': 'model_params', 'type': str, 'default': 'glove'},
         {'name': 'rel_method', 'for': 'train_params', 'type': eval, 'default': score},
         {'name': 'rel_score_obj_scale', 'for': 'train_params', 'type': float, 'default': 1e-3},
         {'name': 'rel_score_penalty', 'for': 'train_params', 'type': float, 'default': 0.5},
         {'name': 'train_dataset_size', 'for': 'train_params', 'type': lambda size: int(size) if size is not None else None, 'default': None},
         {'name': 'truncation', 'for': 'train_params', 'type': float, 'default': -1.0},
         {'name': 'use_cnn', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_cosine_similarity', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_truncated_hinge_loss', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_gradient_clipping', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_lstm', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_max_pooling', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_pointwise_loss', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_pretrained_doc_encoder', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_sequential_sampler', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_single_word_embed_set', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'weight_decay', 'for': 'train_params', 'type': float, 'default': 0.0}]

def name(path, notes):
  if len(notes) == 0: return path
  path_segs = path.split('.json')
  return '_'.join([path_segs[0]] + notes) + '.json'

class MyRabbit(Rabbit):
  def run(self):
    pass

model_to_save = None
experiment = None

def main():
  global model_to_save
  global experiment
  rabbit = MyRabbit(args)
  experiment = Experiment(rabbit.train_params + rabbit.model_params + rabbit.run_params)
  use_pretrained_doc_encoder = rabbit.model_params.use_pretrained_doc_encoder
  use_pointwise_loss = rabbit.train_params.use_pointwise_loss
  query_token_embed_len = rabbit.model_params.query_token_embed_len
  document_token_embed_len = rabbit.model_params.document_token_embed_len
  if not rabbit.run_params.just_caches:
    document_lookup = read_cache('./doc_lookup.json', get_robust_documents)
  num_doc_tokens_to_consider = rabbit.train_params.num_doc_tokens_to_consider
  document_title_to_id = read_cache('./document_title_to_id.json',
                                    lambda: create_id_lookup(document_lookup.keys()))
  if not rabbit.model_params.dont_limit_num_uniq_tokens:
    with open('./60000_most_common_doc.json', 'r') as fh:
      doc_token_set = set(json.load(fh))
    documents, document_token_lookup = read_cache(f'./parsed_docs_{num_doc_tokens_to_consider}_tokens_limit_uniq_toks.json',
                                                  lambda: prepare(document_lookup,
                                                                  document_title_to_id,
                                                                  num_tokens=num_doc_tokens_to_consider,
                                                                  token_set=doc_token_set))
  else:
    documents, document_token_lookup = read_cache(f'./parsed_docs_{num_doc_tokens_to_consider}_tokens_no_pad.json',
                                                  lambda: prepare(document_lookup,
                                                                  document_title_to_id,
                                                                  num_tokens=num_doc_tokens_to_consider))
  if not rabbit.run_params.just_caches:
    train_query_lookup = read_cache('./robust_train_queries.json', get_robust_train_queries)
    train_query_name_to_id = read_cache('./train_query_name_to_id.json',
                                        lambda: create_id_lookup(train_query_lookup.keys()))
  with open('./60000_most_common_query.json', 'r') as fh:
    query_token_set = set(json.load(fh))
  if not rabbit.model_params.dont_limit_num_uniq_tokens:
    train_queries, query_token_lookup = read_cache('./parsed_robust_queries_limit_uniq_toks.json',
                                                   lambda: prepare(train_query_lookup,
                                                                   train_query_name_to_id,
                                                                   token_set=query_token_set))
  else:
    train_queries, query_token_lookup = read_cache('./parsed_robust_queries.json',
                                                   lambda: prepare(train_query_lookup, train_query_name_to_id))
  if rabbit.model_params.frame_as_qa or rabbit.model_params.use_single_word_embed_set:
    query_tok_to_doc_tok = {idx: document_token_lookup.get(query_token) or document_token_lookup['<unk>']
                            for query_token, idx in query_token_lookup.items()}
  else:
    query_tok_to_doc_tok = None
  if rabbit.train_params.train_dataset_size:
    train_data = read_cache(name(f'./robust_train_query_results_tokens_first_{rabbit.train_params.train_dataset_size}.json',
                                 ['limit_uniq_toks'] if not rabbit.model_params.dont_limit_num_uniq_tokens else []),
                            lambda: read_query_result(train_query_name_to_id,
                                                      document_title_to_id,
                                                      train_queries)[:rabbit.train_params.train_dataset_size])
  else:
    train_data = read_cache(name(f'./robust_train_query_results_tokens.json',
                                 ['limit_uniq_toks'] if not rabbit.model_params.dont_limit_num_uniq_tokens else []),
                            lambda: read_query_result(train_query_name_to_id,
                                                      document_title_to_id,
                                                      train_queries))
  q_embed_len = rabbit.model_params.query_token_embed_len
  doc_embed_len = rabbit.model_params.document_token_embed_len
  if q_embed_len == doc_embed_len:
    glove_lookup = get_glove_lookup(embedding_dim=q_embed_len)
    q_glove_lookup = glove_lookup
    doc_glove_lookup = glove_lookup
  else:
    q_glove_lookup = get_glove_lookup(embedding_dim=q_embed_len)
    doc_glove_lookup = get_glove_lookup(embedding_dim=doc_embed_len)
  num_query_tokens = len(query_token_lookup)
  num_doc_tokens = len(document_token_lookup)
  doc_encoder = None
  if use_pretrained_doc_encoder:
    doc_encoder, document_token_embeds = get_doc_encoder_and_embeddings(document_token_lookup,
                                                                        rabbit.model_params.only_use_last_out)
    query_token_embeds_init = from_doc_to_query_embeds(document_token_embeds,
                                                       document_token_lookup,
                                                       query_token_lookup)
    if not rabbit.train_params.dont_freeze_pretrained_doc_encoder:
      dont_update(doc_encoder)
  else:
    document_token_embeds = init_embedding(doc_glove_lookup,
                                           document_token_lookup,
                                           num_doc_tokens,
                                           document_token_embed_len)
    if rabbit.model_params.use_single_word_embed_set:
      query_token_embeds_init = document_token_embeds
    else:
      query_token_embeds_init = init_embedding(q_glove_lookup,
                                               query_token_lookup,
                                               num_query_tokens,
                                               query_token_embed_len)
  if not rabbit.train_params.dont_freeze_word_embeds:
    dont_update(document_token_embeds)
    dont_update(query_token_embeds_init)
  else:
    do_update(document_token_embeds)
    do_update(query_token_embeds_init)
  if rabbit.train_params.add_rel_score:
    query_token_embeds, additive = get_additive_regularized_embeds(query_token_embeds_init)
    rel_score = RelScore(query_token_embeds, document_token_embeds, rabbit.model_params, rabbit.train_params)
  else:
    query_token_embeds = query_token_embeds_init
    additive = None
    rel_score = None
  test_query_lookup = read_cache('./robust_test_queries.json',
                                 get_robust_test_queries)
  test_query_name_document_title_rels = read_cache('./robust_rels.json',
                                                   get_robust_rels)
  test_query_name_to_id = read_cache('./test_query_name_to_id.json',
                                     lambda: create_id_lookup(test_query_lookup.keys()))
  test_queries, __ = read_cache(name('./parsed_test_robust_queries.json',
                                     ['limit_uniq_toks'] if not rabbit.model_params.dont_limit_num_uniq_tokens else []),
                                lambda: prepare(test_query_lookup,
                                                test_query_name_to_id,
                                                token_lookup=query_token_lookup))
  test_data = read_cache(name('./parsed_robust_rels.json',
                              ['limit_uniq_toks'] if not rabbit.model_params.dont_limit_num_uniq_tokens else []),
                         lambda: process_rels(test_query_name_document_title_rels,
                                              document_title_to_id,
                                              test_query_name_to_id,
                                              test_queries))
  if use_pointwise_loss:
    normalized_train_data = read_cache(name('./normalized_train_query_data.json',
                                            ['limit_uniq_toks'] if not rabbit.model_params.dont_limit_num_uniq_tokens else []),
                                       lambda: normalize_scores_query_wise(train_data))
    train_dl = build_query_dataloader(documents,
                                      normalized_train_data[:rabbit.train_params.train_dataset_size],
                                      rabbit.train_params.batch_size,
                                      rel_method=rabbit.train_params.rel_method,
                                      num_doc_tokens=num_doc_tokens_to_consider,
                                      cache=name('./pointwise_train_ranking.json',
                                                 ['limit_uniq_toks'] if not rabbit.model_params.dont_limit_num_uniq_tokens else []),
                                      limit=10,
                                      query_tok_to_doc_tok=query_tok_to_doc_tok,
                                      use_sequential_sampler=rabbit.train_params.use_sequential_sampler)
    test_dl = build_query_dataloader(documents,
                                     test_data,
                                     rabbit.train_params.batch_size,
                                     rel_method=rabbit.train_params.rel_method,
                                     num_doc_tokens=num_doc_tokens_to_consider,
                                     cache=name('./pointwise_test_ranking.json',
                                                ['limit_uniq_toks'] if not rabbit.model_params.dont_limit_num_uniq_tokens else []),
                                     query_tok_to_doc_tok=query_tok_to_doc_tok,
                                     use_sequential_sampler=rabbit.train_params.use_sequential_sampler)
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
                                               num_neg_samples=rabbit.train_params.num_neg_samples,
                                               num_doc_tokens=num_doc_tokens_to_consider,
                                               cache=name('./pairwise_train_ranking.json',
                                                          ['limit_uniq_toks'] if not rabbit.model_params.dont_limit_num_uniq_tokens else []),
                                               limit=10,
                                               query_tok_to_doc_tok=query_tok_to_doc_tok,
                                               use_sequential_sampler=rabbit.train_params.use_sequential_sampler)
    test_dl = build_query_pairwise_dataloader(documents,
                                              test_data,
                                              rabbit.train_params.batch_size,
                                              rel_method=rabbit.train_params.rel_method,
                                              num_neg_samples=0,
                                              num_doc_tokens=num_doc_tokens_to_consider,
                                              cache=name('./pairwise_test_ranking.json',
                                                         ['limit_uniq_toks'] if not rabbit.model_params.dont_limit_num_uniq_tokens else []),
                                              query_tok_to_doc_tok=query_tok_to_doc_tok,
                                              use_sequential_sampler=rabbit.train_params.use_sequential_sampler)
    model = PairwiseScorer(query_token_embeds,
                           document_token_embeds,
                           doc_encoder,
                           rabbit.model_params,
                           rabbit.train_params)
  train_ranking_dataset = RankingDataset(documents,
                                         train_dl.dataset.rankings,
                                         num_doc_tokens=num_doc_tokens_to_consider,
                                         query_tok_to_doc_tok=query_tok_to_doc_tok)
  test_ranking_candidates = read_cache('./test_ranking_candidates.json',
                                       read_query_test_rankings)
  lookup_by_title = lambda title: document_title_to_id.get(title) or 0
  test_ranking_candidates = _.map_values(test_ranking_candidates,
                                         lambda candidate_names: _.map_(candidate_names,
                                                                        lookup_by_title))
  test_ranking_candidates = _.map_keys(test_ranking_candidates,
                                       lambda ranking, query_name: str(test_queries[test_query_name_to_id[query_name]])[1:-1])
  test_ranking_dataset = RankingDataset(documents,
                                        test_ranking_candidates,
                                        test_dl.dataset.rankings,
                                        num_doc_tokens=num_doc_tokens_to_consider,
                                        query_tok_to_doc_tok=query_tok_to_doc_tok)
  valid_dl = build_query_pairwise_dataloader(documents,
                                             test_data[:rabbit.train_params.batch_size],
                                             rabbit.train_params.batch_size)
  if rabbit.train_params.memorize_test:
    train_dl = test_dl
    train_ranking_dataset = test_ranking_dataset
  model_data = DataBunch(train_dl,
                         valid_dl,
                         test_dl,
                         collate_fn=collate_query_samples if use_pointwise_loss else collate_query_pairwise_samples,
                         device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
  multi_objective_model = MultiObjective(model, rabbit.train_params, rel_score, additive)
  model_to_save = multi_objective_model
  if not rabbit.run_params.just_caches:
    del document_lookup
    del train_query_lookup
  del query_token_lookup
  del document_token_lookup
  del test_query_lookup
  del train_queries
  del test_queries
  del glove_lookup
  train_model(multi_objective_model,
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
      torch.save(model_to_save.state_dict(), './model_save_debug' + str(experiment.model_name))
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
  ipdb.post_mortem(tb)
