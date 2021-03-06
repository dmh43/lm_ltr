import pickle
import json
import random
import time
from heapq import nlargest, nsmallest
from operator import itemgetter
from functools import reduce
from collections import defaultdict

import pydash as _
import torch
import torch.nn as nn
from fastai.basic_data import DataBunch, DeviceDataLoader
from fastai import to_device
from fastai.text import Tokenizer
from progressbar import progressbar

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, Sampler, SequentialSampler

from lm_ltr.embedding_loaders import get_glove_lookup, init_embedding, extend_token_lookup, from_doc_to_query_embeds, get_additive_regularized_embeds
from lm_ltr.fetchers import get_raw_documents, get_supervised_raw_data, get_weak_raw_data, read_or_cache, read_cache, get_robust_documents, get_robust_train_queries, get_robust_eval_queries, get_robust_rels, read_query_result, read_query_test_rankings, read_from_file, get_robust_documents_with_titles, get_ranker_query_str_to_pairwise_bins, get_ranker_query_str_to_rankings
from lm_ltr.pointwise_scorer import PointwiseScorer
from lm_ltr.pairwise_scorer import PairwiseScorer
from lm_ltr.preprocessing import preprocess_texts, all_ones, score, inv_log_rank, inv_rank, exp_score, collate_query_samples, collate_query_pairwise_samples, prepare, prepare_fs, create_id_lookup, normalize_scores_query_wise, process_rels, get_normalized_score_lookup, process_raw_candidates
from lm_ltr.data_wrappers import build_query_dataloader, build_query_pairwise_dataloader, RankingDataset, SequentialSamplerWithLimit
from lm_ltr.train_model import train_model
from lm_ltr.pretrained import get_doc_encoder_and_embeddings
from lm_ltr.utils import dont_update, do_update, name
from lm_ltr.multi_objective import MultiObjective
from lm_ltr.rel_score import RelScore
from lm_ltr.regularization import Regularization
from lm_ltr.snorkel_helper import Snorkeller
from lm_ltr.globals import RANKER_NAME_TO_SUFFIX
from lm_ltr.influence import get_num_neg_influences, calc_test_hvps, calc_influence, calc_dataset_influence

from rabbit_ml.rabbit_ml import Rabbit
from rabbit_ml.rabbit_ml.arg_parsers import list_arg, optional_arg
from rabbit_ml.rabbit_ml.experiment import Experiment

args =  [{'name': 'ablation', 'for': 'model_params', 'type': list_arg(str), 'default': []},
         {'name': 'add_rel_score', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'append_hadamard', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'append_difference', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'batch_size', 'for': 'train_params', 'type': int, 'default': 512},
         {'name': 'bin_rankings', 'for': 'train_params', 'type': optional_arg(int), 'default': None},
         {'name': 'calc_influence', 'for': 'run_params', 'type': 'flag', 'default': False},
         {'name': 'cheat', 'for': 'run_params', 'type': bool, 'default': False},
         {'name': 'comments', 'for': 'run_params', 'type': str, 'default': ''},
         {'name': 'document_token_embed_len', 'for': 'model_params', 'type': int, 'default': 100},
         {'name': 'document_token_embedding_set', 'for': 'model_params', 'type': str, 'default': 'glove'},
         {'name': 'dont_freeze_pretrained_doc_encoder', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'dont_freeze_word_embeds', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'dont_include_normalized_score', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'dont_limit_num_uniq_tokens', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'dont_smooth', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'dropout_keep_prob', 'for': 'train_params', 'type': float, 'default': 0.8},
         {'name': 'drop_val_loss_calc', 'for': 'run_params', 'type': 'flag', 'default': False},
         {'name': 'dont_include_titles', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'dont_use_bow', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'num_to_drop_in_ranking', 'for': 'train_params', 'type': int, 'default': 0},
         {'name': 'fine_tune_on_val', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'frame_as_qa', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'freeze_all_but_last_for_influence', 'for': 'run_params', 'type': 'flag', 'default': False},
         {'name': 'gradient_clipping_norm', 'for': 'train_params', 'type': float, 'default': 0.1},
         {'name': 'hidden_layer_sizes', 'for': 'model_params', 'type': list_arg(int), 'default': [128, 64, 16]},
         {'name': 'just_caches', 'for': 'run_params', 'type': 'flag', 'default': False},
         {'name': 'influences_path', 'for': 'run_params', 'type': str, 'default': 'most_hurtful.json'},
         {'name': 'influence_thresh', 'for': 'train_params', 'type': float, 'default': -1500.0},
         {'name': 'keep_top_uniq_terms', 'for': 'model_params', 'type': optional_arg(int), 'default': None},
         {'name': 'learning_rate', 'for': 'train_params', 'type': float, 'default': 1e-3},
         {'name': 'load_model', 'for': 'run_params', 'type': 'flag', 'default': False},
         {'name': 'load_influences', 'for': 'run_params', 'type': 'flag', 'default': False},
         {'name': 'load_path', 'for': 'run_params', 'type': optional_arg(str), 'default': None},
         {'name': 'lstm_hidden_size', 'for': 'model_params', 'type': int, 'default': 100},
         {'name': 'max_cg_iters', 'for': 'run_params', 'type': optional_arg(int), 'default': None},
         {'name': 'margin', 'for': 'train_params', 'type': float, 'default': 1.0},
         {'name': 'max_iter', 'for': 'train_params', 'type': optional_arg(int), 'default': None},
         {'name': 'memorize_test', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'nce_sample_mul_rel_score', 'for': 'train_params', 'type': int, 'default': 5},
         {'name': 'num_doc_tokens_to_consider', 'for': 'train_params', 'type': int, 'default': 100},
         {'name': 'num_epochs', 'for': 'train_params', 'type': int, 'default': 1},
         {'name': 'num_neg_samples', 'for': 'train_params', 'type': int, 'default': 0},
         {'name': 'num_pos_tokens_rel_score', 'for': 'train_params', 'type': int, 'default': 20},
         {'name': 'num_to_rank', 'for': 'run_params', 'type': int, 'default': 1000},
         {'name': 'num_train_queries', 'for': 'train_params', 'type': optional_arg(int), 'default': None},
         {'name': 'only_use_last_out', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'optimizer', 'for': 'train_params', 'type': str, 'default': 'adam'},
         {'name': 'query_token_embed_len', 'for': 'model_params', 'type': int, 'default': 100},
         {'name': 'query_token_embedding_set', 'for': 'model_params', 'type': str, 'default': 'glove'},
         {'name': 'ranking_set', 'for': 'train_params', 'type': str, 'default': 'qml'},
         {'name': 'rel_method', 'for': 'train_params', 'type': eval, 'default': score},
         {'name': 'rel_score_obj_scale', 'for': 'train_params', 'type': float, 'default': 0.1},
         {'name': 'rel_score_penalty', 'for': 'train_params', 'type': float, 'default': 5e-4},
         {'name': 'swap_labels', 'for': 'train_params', 'type': float, 'default': 0.0},
         {'name': 'num_snorkel_train_queries', 'for': 'train_params', 'type': int, 'default': 10000},
         {'name': 'record_every_n', 'for': 'run_params', 'type': int, 'default': 10000},
         {'name': 'truncation', 'for': 'train_params', 'type': float, 'default': -1.0},
         {'name': 'use_batch_norm', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_bce_loss', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_cnn', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_cosine_similarity', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_cyclical_lr', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_dense', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_doc_out', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_truncated_hinge_loss', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_glove', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_gradient_clipping', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_gauss_newton', 'for': 'run_params', 'type': 'flag', 'default': False},
         {'name': 'use_l1_loss', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_large_embed', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_layer_norm', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_lstm', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_label_smoothing', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_max_pooling', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_noise_aware_loss', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_pointwise_loss', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_pretrained_doc_encoder', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_scipy', 'for': 'run_params', 'type': 'flag', 'default': False},
         {'name': 'use_sequential_sampler', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_single_word_embed_set', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_softrank_influence', 'for': 'run_params', 'type': 'flag', 'default': False},
         {'name': 'use_weighted_loss', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'use_word2vec', 'for': 'model_params', 'type': 'flag', 'default': False},
         {'name': 'use_variable_loss', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'weight_decay', 'for': 'train_params', 'type': float, 'default': 0.0},
         {'name': 'weight_influence', 'for': 'train_params', 'type': 'flag', 'default': False},
         {'name': 'word_level_do_kp', 'for': 'train_params', 'type': float, 'default': 1.0}]

class MyRabbit(Rabbit):
  def run(self):
    pass

model_to_save = None
experiment = None
rabbit = None

def main():
  global model_to_save
  global experiment
  global rabbit
  rabbit = MyRabbit(args)
  if rabbit.model_params.dont_limit_num_uniq_tokens: raise NotImplementedError()
  if rabbit.model_params.frame_as_qa: raise NotImplementedError
  if rabbit.run_params.drop_val_loss_calc: raise NotImplementedError
  if rabbit.run_params.use_softrank_influence and not rabbit.run_params.freeze_all_but_last_for_influence: raise NotImplementedError
  if rabbit.train_params.weight_influence: raise NotImplementedError
  experiment = Experiment(rabbit.train_params + rabbit.model_params + rabbit.run_params)
  print('Model name:', experiment.model_name)
  use_pretrained_doc_encoder = rabbit.model_params.use_pretrained_doc_encoder
  use_pointwise_loss = rabbit.train_params.use_pointwise_loss
  query_token_embed_len = rabbit.model_params.query_token_embed_len
  document_token_embed_len = rabbit.model_params.document_token_embed_len
  _names = []
  if not rabbit.model_params.dont_include_titles:
    _names.append('with_titles')
  if rabbit.train_params.num_doc_tokens_to_consider != -1:
    _names.append('num_doc_toks_' + str(rabbit.train_params.num_doc_tokens_to_consider))
  if not rabbit.run_params.just_caches:
    if rabbit.model_params.dont_include_titles:
      document_lookup = read_cache(name('./doc_lookup.json', _names), get_robust_documents)
    else:
      document_lookup = read_cache(name('./doc_lookup.json', _names), get_robust_documents_with_titles)
  num_doc_tokens_to_consider = rabbit.train_params.num_doc_tokens_to_consider
  document_title_to_id = read_cache('./document_title_to_id.json',
                                    lambda: create_id_lookup(document_lookup.keys()))
  with open('./caches/106756_most_common_doc.json', 'r') as fh:
    doc_token_set = set(json.load(fh))
    tokenizer = Tokenizer()
    tokenized = set(sum(tokenizer.process_all(list(get_robust_eval_queries().values())), []))
    doc_token_set = doc_token_set.union(tokenized)
  use_bow_model = not any([rabbit.model_params[attr] for attr in ['use_doc_out',
                                                                  'use_cnn',
                                                                  'use_lstm',
                                                                  'use_pretrained_doc_encoder']])
  use_bow_model = use_bow_model and not rabbit.model_params.dont_use_bow
  if use_bow_model:
    documents, document_token_lookup = read_cache(name(f'./docs_fs_tokens_limit_uniq_toks_qrels_and_106756.pkl',
                                                       _names),
                                                lambda: prepare_fs(document_lookup,
                                                                   document_title_to_id,
                                                                   num_tokens=num_doc_tokens_to_consider,
                                                                   token_set=doc_token_set))
    if rabbit.model_params.keep_top_uniq_terms is not None:
      documents = [dict(nlargest(rabbit.model_params.keep_top_uniq_terms,
                                 _.to_pairs(doc),
                                 itemgetter(1)))
                   for doc in documents]
  else:
    documents, document_token_lookup = read_cache(name(f'./parsed_docs_{num_doc_tokens_to_consider}_tokens_limit_uniq_toks_qrels_and_106756.json',
                                                     _names),
                                                lambda: prepare(document_lookup,
                                                                document_title_to_id,
                                                                num_tokens=num_doc_tokens_to_consider,
                                                                token_set=doc_token_set))
  if not rabbit.run_params.just_caches:
    train_query_lookup = read_cache('./robust_train_queries.json', get_robust_train_queries)
    train_query_name_to_id = read_cache('./train_query_name_to_id.json',
                                        lambda: create_id_lookup(train_query_lookup.keys()))
  train_queries, query_token_lookup = read_cache('./parsed_robust_queries_dict.json',
                                                 lambda: prepare(train_query_lookup,
                                                                 train_query_name_to_id,
                                                                 token_lookup=document_token_lookup,
                                                                 token_set=doc_token_set,
                                                                 drop_if_any_unk=True))
  query_tok_to_doc_tok = {idx: document_token_lookup.get(query_token) or document_token_lookup['<unk>']
                          for query_token, idx in query_token_lookup.items()}
  names = [RANKER_NAME_TO_SUFFIX[rabbit.train_params.ranking_set]]
  if rabbit.train_params.use_pointwise_loss or not rabbit.run_params.just_caches:
    train_data = read_cache(name('./robust_train_query_results_tokens_qrels_and_106756.json', names),
                            lambda: read_query_result(train_query_name_to_id,
                                                      document_title_to_id,
                                                      train_queries,
                                                      path='./indri/query_result' + RANKER_NAME_TO_SUFFIX[rabbit.train_params.ranking_set]))
  else:
    train_data = []
  q_embed_len = rabbit.model_params.query_token_embed_len
  doc_embed_len = rabbit.model_params.document_token_embed_len
  if rabbit.model_params.append_difference or rabbit.model_params.append_hadamard:
    assert q_embed_len == doc_embed_len, 'Must use same size doc and query embeds when appending diff or hadamard'
  if q_embed_len == doc_embed_len:
    glove_lookup = get_glove_lookup(embedding_dim=q_embed_len,
                                    use_large_embed=rabbit.model_params.use_large_embed,
                                    use_word2vec=rabbit.model_params.use_word2vec)
    q_glove_lookup = glove_lookup
    doc_glove_lookup = glove_lookup
  else:
    q_glove_lookup = get_glove_lookup(embedding_dim=q_embed_len,
                                      use_large_embed=rabbit.model_params.use_large_embed,
                                      use_word2vec=rabbit.model_params.use_word2vec)
    doc_glove_lookup = get_glove_lookup(embedding_dim=doc_embed_len,
                                        use_large_embed=rabbit.model_params.use_large_embed,
                                        use_word2vec=rabbit.model_params.use_word2vec)
  num_query_tokens = len(query_token_lookup)
  num_doc_tokens = len(document_token_lookup)
  doc_encoder = None
  if use_pretrained_doc_encoder or rabbit.model_params.use_doc_out:
    doc_encoder, document_token_embeds = get_doc_encoder_and_embeddings(document_token_lookup,
                                                                        rabbit.model_params.only_use_last_out)
    if rabbit.model_params.use_glove:
      query_token_embeds_init = init_embedding(q_glove_lookup,
                                               query_token_lookup,
                                               num_query_tokens,
                                               query_token_embed_len)
    else:
      query_token_embeds_init = from_doc_to_query_embeds(document_token_embeds,
                                                         document_token_lookup,
                                                         query_token_lookup)
    if not rabbit.train_params.dont_freeze_pretrained_doc_encoder:
      dont_update(doc_encoder)
    if rabbit.model_params.use_doc_out:
      doc_encoder = None
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
  eval_query_lookup = get_robust_eval_queries()
  eval_query_name_document_title_rels = get_robust_rels()
  test_query_names = []
  val_query_names = []
  for query_name in eval_query_lookup:
    if len(val_query_names) >= 50: test_query_names.append(query_name)
    else: val_query_names.append(query_name)
  test_query_name_document_title_rels = _.pick(eval_query_name_document_title_rels, test_query_names)
  test_query_lookup = _.pick(eval_query_lookup, test_query_names)
  test_query_name_to_id = create_id_lookup(test_query_lookup.keys())
  test_queries, __ = prepare(test_query_lookup,
                             test_query_name_to_id,
                             token_lookup=query_token_lookup)
  eval_ranking_candidates = read_query_test_rankings('./indri/query_result_test' + RANKER_NAME_TO_SUFFIX[rabbit.train_params.ranking_set])
  test_candidates_data = read_query_result(test_query_name_to_id,
                                           document_title_to_id,
                                           dict(zip(range(len(test_queries)),
                                                    test_queries)),
                                           path='./indri/query_result_test' + RANKER_NAME_TO_SUFFIX[rabbit.train_params.ranking_set])
  test_ranking_candidates = process_raw_candidates(test_query_name_to_id,
                                                   test_queries,
                                                   document_title_to_id,
                                                   test_query_names,
                                                   eval_ranking_candidates)
  test_data = process_rels(test_query_name_document_title_rels,
                           document_title_to_id,
                           test_query_name_to_id,
                           test_queries)
  val_query_name_document_title_rels = _.pick(eval_query_name_document_title_rels, val_query_names)
  val_query_lookup = _.pick(eval_query_lookup, val_query_names)
  val_query_name_to_id = create_id_lookup(val_query_lookup.keys())
  val_queries, __ = prepare(val_query_lookup,
                            val_query_name_to_id,
                            token_lookup=query_token_lookup)
  val_candidates_data = read_query_result(val_query_name_to_id,
                                          document_title_to_id,
                                          dict(zip(range(len(val_queries)),
                                                    val_queries)),
                                          path='./indri/query_result_test' + RANKER_NAME_TO_SUFFIX[rabbit.train_params.ranking_set])
  val_ranking_candidates = process_raw_candidates(val_query_name_to_id,
                                                  val_queries,
                                                  document_title_to_id,
                                                  val_query_names,
                                                  eval_ranking_candidates)
  val_data = process_rels(val_query_name_document_title_rels,
                          document_title_to_id,
                          val_query_name_to_id,
                          val_queries)
  train_normalized_score_lookup = read_cache(name('./train_normalized_score_lookup.pkl', names),
                                             lambda: get_normalized_score_lookup(train_data))
  test_normalized_score_lookup = get_normalized_score_lookup(test_candidates_data)
  val_normalized_score_lookup = get_normalized_score_lookup(val_candidates_data)
  if use_pointwise_loss:
    normalized_train_data = read_cache(name('./normalized_train_query_data_qrels_and_106756.json', names),
                                       lambda: normalize_scores_query_wise(train_data))
    collate_fn = lambda samples: collate_query_samples(samples,
                                                       use_bow_model=use_bow_model,
                                                       use_dense=rabbit.model_params.use_dense)
    train_dl = build_query_dataloader(documents,
                                      normalized_train_data,
                                      rabbit.train_params,
                                      rabbit.model_params,
                                      cache=name('train_ranking_qrels_and_106756.json', names),
                                      limit=10,
                                      query_tok_to_doc_tok=query_tok_to_doc_tok,
                                      normalized_score_lookup=train_normalized_score_lookup,
                                      use_bow_model=use_bow_model,
                                      collate_fn=collate_fn,
                                      is_test=False)
    test_dl = build_query_dataloader(documents,
                                     test_data,
                                     rabbit.train_params,
                                     rabbit.model_params,
                                     query_tok_to_doc_tok=query_tok_to_doc_tok,
                                     normalized_score_lookup=test_normalized_score_lookup,
                                     use_bow_model=use_bow_model,
                                     collate_fn=collate_fn,
                                     is_test=True)
    val_dl = build_query_dataloader(documents,
                                    val_data,
                                    rabbit.train_params,
                                    rabbit.model_params,
                                    query_tok_to_doc_tok=query_tok_to_doc_tok,
                                    normalized_score_lookup=val_normalized_score_lookup,
                                    use_bow_model=use_bow_model,
                                    collate_fn=collate_fn,
                                    is_test=True)
    model = PointwiseScorer(query_token_embeds,
                            document_token_embeds,
                            doc_encoder,
                            rabbit.model_params,
                            rabbit.train_params)
  else:
    if rabbit.train_params.use_noise_aware_loss:
      ranker_query_str_to_rankings = get_ranker_query_str_to_rankings(train_query_name_to_id,
                                                                      document_title_to_id,
                                                                      train_queries,
                                                                      limit=rabbit.train_params.num_snorkel_train_queries)
      query_names = reduce(lambda acc, query_to_ranking: acc.intersection(set(query_to_ranking.keys())) if len(acc) != 0 else set(query_to_ranking.keys()),
                           ranker_query_str_to_rankings.values(),
                           set())
      all_ranked_lists_by_ranker = _.map_values(ranker_query_str_to_rankings,
                                                lambda query_to_ranking: [query_to_ranking[query]
                                                                          for query in query_names])
      ranker_query_str_to_pairwise_bins = get_ranker_query_str_to_pairwise_bins(train_query_name_to_id,
                                                                                document_title_to_id,
                                                                                train_queries,
                                                                                limit=rabbit.train_params.num_train_queries)
      snorkeller = Snorkeller(ranker_query_str_to_pairwise_bins)
      snorkeller.train(all_ranked_lists_by_ranker)
      calc_marginals = snorkeller.calc_marginals
    else:
      calc_marginals = None
    collate_fn = lambda samples: collate_query_pairwise_samples(samples,
                                                                use_bow_model=use_bow_model,
                                                                calc_marginals=calc_marginals,
                                                                use_dense=rabbit.model_params.use_dense)
    if rabbit.run_params.load_influences:
      try:
        with open(rabbit.run_params.influences_path) as fh:
          pairs_to_flip = defaultdict(set)
          for pair, influence in json.load(fh):
            if rabbit.train_params.use_pointwise_loss:
              condition = True
            else:
              condition = influence < rabbit.train_params.influence_thresh
            if condition:
              query = tuple(pair[1])
              pairs_to_flip[query].add(tuple(pair[0]))
      except FileNotFoundError:
        pairs_to_flip = None
    else:
      pairs_to_flip = None
    train_dl = build_query_pairwise_dataloader(documents,
                                               train_data,
                                               rabbit.train_params,
                                               rabbit.model_params,
                                               pairs_to_flip=pairs_to_flip,
                                               cache=name('train_ranking_qrels_and_106756.json', names),
                                               limit=10,
                                               query_tok_to_doc_tok=query_tok_to_doc_tok,
                                               normalized_score_lookup=train_normalized_score_lookup,
                                               use_bow_model=use_bow_model,
                                               collate_fn=collate_fn,
                                               is_test=False)
    test_dl = build_query_pairwise_dataloader(documents,
                                              test_data,
                                              rabbit.train_params,
                                              rabbit.model_params,
                                              query_tok_to_doc_tok=query_tok_to_doc_tok,
                                              normalized_score_lookup=test_normalized_score_lookup,
                                              use_bow_model=use_bow_model,
                                              collate_fn=collate_fn,
                                              is_test=True)
    val_dl = build_query_pairwise_dataloader(documents,
                                             val_data,
                                             rabbit.train_params,
                                             rabbit.model_params,
                                             query_tok_to_doc_tok=query_tok_to_doc_tok,
                                             normalized_score_lookup=val_normalized_score_lookup,
                                             use_bow_model=use_bow_model,
                                             collate_fn=collate_fn,
                                             is_test=True)
    val_rel_dl = build_query_pairwise_dataloader(documents,
                                                 val_data,
                                                 rabbit.train_params,
                                                 rabbit.model_params,
                                                 query_tok_to_doc_tok=query_tok_to_doc_tok,
                                                 normalized_score_lookup=val_normalized_score_lookup,
                                                 use_bow_model=use_bow_model,
                                                 collate_fn=collate_fn,
                                                 is_test=True,
                                                 rel_vs_irrel=True,
                                                 candidates=val_ranking_candidates,
                                                 num_to_rank=rabbit.run_params.num_to_rank)
    model = PairwiseScorer(query_token_embeds,
                           document_token_embeds,
                           doc_encoder,
                           rabbit.model_params,
                           rabbit.train_params,
                           use_bow_model=use_bow_model)
  train_ranking_dataset = RankingDataset(documents,
                                         train_dl.dataset.rankings,
                                         rabbit.train_params,
                                         rabbit.model_params,
                                         rabbit.run_params,
                                         query_tok_to_doc_tok=query_tok_to_doc_tok,
                                         normalized_score_lookup=train_normalized_score_lookup,
                                         use_bow_model=use_bow_model,
                                         use_dense=rabbit.model_params.use_dense)
  test_ranking_dataset = RankingDataset(documents,
                                        test_ranking_candidates,
                                        rabbit.train_params,
                                        rabbit.model_params,
                                        rabbit.run_params,
                                        relevant=test_dl.dataset.rankings,
                                        query_tok_to_doc_tok=query_tok_to_doc_tok,
                                        cheat=rabbit.run_params.cheat,
                                        normalized_score_lookup=test_normalized_score_lookup,
                                        use_bow_model=use_bow_model,
                                        use_dense=rabbit.model_params.use_dense)
  val_ranking_dataset = RankingDataset(documents,
                                       val_ranking_candidates,
                                       rabbit.train_params,
                                       rabbit.model_params,
                                       rabbit.run_params,
                                       relevant=val_dl.dataset.rankings,
                                       query_tok_to_doc_tok=query_tok_to_doc_tok,
                                       cheat=rabbit.run_params.cheat,
                                       normalized_score_lookup=val_normalized_score_lookup,
                                       use_bow_model=use_bow_model,
                                       use_dense=rabbit.model_params.use_dense)
  if rabbit.train_params.memorize_test:
    train_dl = test_dl
    train_ranking_dataset = test_ranking_dataset
  model_data = DataBunch(train_dl,
                         val_rel_dl,
                         test_dl,
                         collate_fn=collate_fn,
                         device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
  multi_objective_model = MultiObjective(model, rabbit.train_params, rel_score, additive)
  model_to_save = multi_objective_model
  if rabbit.train_params.memorize_test:
    try: del train_data
    except: pass
  if not rabbit.run_params.just_caches:
    del document_lookup
    del train_query_lookup
  del query_token_lookup
  del document_token_lookup
  del train_queries
  try:
    del glove_lookup
  except UnboundLocalError:
    del q_glove_lookup
    del doc_glove_lookup
  if rabbit.run_params.load_model:
    try:
      multi_objective_model.load_state_dict(torch.load(rabbit.run_params.load_path))
    except RuntimeError:
      dp = nn.DataParallel(multi_objective_model)
      dp.load_state_dict(torch.load(rabbit.run_params.load_path))
      multi_objective_model = dp.module
  else:
    train_model(multi_objective_model,
                model_data,
                train_ranking_dataset,
                val_ranking_dataset,
                test_ranking_dataset,
                rabbit.train_params,
                rabbit.model_params,
                rabbit.run_params,
                experiment)
  if rabbit.train_params.fine_tune_on_val:
    fine_tune_model_data = DataBunch(val_rel_dl,
                                     val_rel_dl,
                                     test_dl,
                                     collate_fn=collate_fn,
                                     device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    train_model(multi_objective_model,
                fine_tune_model_data,
                val_ranking_dataset,
                val_ranking_dataset,
                test_ranking_dataset,
                rabbit.train_params,
                rabbit.model_params,
                rabbit.run_params,
                experiment,
                load_path=rabbit.run_params.load_path)
  multi_objective_model.eval()
  device = model_data.device
  gpu_multi_objective_model = multi_objective_model.to(device)
  if rabbit.run_params.calc_influence:
    if rabbit.run_params.freeze_all_but_last_for_influence:
      last_layer_idx = _.find_last_index(multi_objective_model.model.pointwise_scorer.layers,
                                         lambda layer: isinstance(layer, nn.Linear))
      to_last_layer = lambda x: gpu_multi_objective_model(*x, to_idx=last_layer_idx)
      last_layer = gpu_multi_objective_model.model.pointwise_scorer.layers[last_layer_idx]
      diff_wrt = [p for p in last_layer.parameters() if p.requires_grad]
    else:
      diff_wrt = None
    test_hvps = calc_test_hvps(multi_objective_model.loss,
                               gpu_multi_objective_model,
                               DeviceDataLoader(train_dl, device, collate_fn=collate_fn),
                               val_rel_dl,
                               rabbit.run_params,
                               diff_wrt=diff_wrt,
                               show_progress=True,
                               use_softrank_influence=rabbit.run_params.use_softrank_influence)
    influences = []
    if rabbit.train_params.use_pointwise_loss:
      num_real_samples = len(train_dl.dataset)
    else:
      num_real_samples = train_dl.dataset._num_pos_pairs
    if rabbit.run_params.freeze_all_but_last_for_influence:
      _sampler = SequentialSamplerWithLimit(train_dl.dataset, num_real_samples)
      _batch_sampler = BatchSampler(_sampler, rabbit.train_params.batch_size, False)
      _dl = DataLoader(train_dl.dataset, batch_sampler=_batch_sampler, collate_fn=collate_fn)
      sequential_train_dl = DeviceDataLoader(_dl, device, collate_fn=collate_fn)
      influences = calc_dataset_influence(gpu_multi_objective_model,
                                          to_last_layer,
                                          sequential_train_dl,
                                          test_hvps,
                                          sum_p=True).tolist()
    else:
      for i in progressbar(range(num_real_samples)):
        train_sample = train_dl.dataset[i]
        x, labels = to_device(collate_fn([train_sample]), device)
        device_train_sample = (x, labels.squeeze())
        influences.append(calc_influence(multi_objective_model.loss,
                                         gpu_multi_objective_model,
                                         device_train_sample,
                                         test_hvps,
                                         diff_wrt=diff_wrt).sum().tolist())
    with open(rabbit.run_params.influences_path, 'w+') as fh:
      json.dump([[train_dl.dataset[idx][1], influence] for idx, influence in enumerate(influences)], fh)

if __name__ == "__main__":
  import ipdb
  import traceback
  import sys

  try:
    main()
  except: # pylint: disable=bare-except
    if not rabbit.run_params.load_model:
      if model_to_save and input("save?") == 'y':
        torch.save(model_to_save.state_dict(),
                   './model_save_debug' + str(experiment.model_name))
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    ipdb.post_mortem(tb)
