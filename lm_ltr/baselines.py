from six.moves import xrange
import pydash as _
from itertools import combinations
import json
from collections import Counter, defaultdict
from operator import itemgetter
from heapq import nlargest

from gensim.summarization.bm25 import BM25
from fastai.text import Tokenizer, fix_html, spec_add_spaces, rm_useless_spaces
import numpy as np
from progressbar import progressbar
import torch

from lm_ltr.fetchers import read_cache, get_robust_test_queries, get_robust_rels, get_robust_documents
from lm_ltr.preprocessing import create_id_lookup, handle_caps
from lm_ltr.embedding_loaders import get_glove_lookup


def rank_bm25(bm25, q, average_idf):
  return top_k(lambda doc_id: bm25.get_score(q, doc_id, average_idf),
               range(bm25.corpus_size),
               k=10)

def _encode_glove(glove, tokens):
  vec = torch.sum(torch.stack([glove[token] for token in tokens if token in glove]).cuda(), 0)
  return vec / torch.norm(vec)

def _encode_glove_fs(glove, doc_fs):
  vec = torch.sum(torch.stack([cnt * glove[token]
                               for token, cnt in doc_fs.items() if token in glove and cnt > 0]).cuda(), 0)
  return vec

def rank_glove(glove, encoded_docs, query, k=10):
  topk_scores, topk_idxs = torch.topk(torch.sum(encoded_docs * _encode_glove(glove, query), 1),
                                      k=k)
  sorted_scores, sort_idxs = torch.sort(topk_scores, descending=True)
  return topk_idxs[sort_idxs].tolist()

def _get_rel_lm(docs_lms, qml_ranking, q, smooth=0.5):
  query_lm = defaultdict(lambda: -np.inf,
                         {q_term: np.log(cnt / len(q)) for q_term, cnt in Counter(q).items()})
  rel_lm = defaultdict(lambda: -np.inf)
  for doc_id in qml_ranking:
    q_prob_in_doc = np.sum([docs_lms[doc_id][q_term] for q_term in q])
    for key, val in docs_lms[doc_id].items():
      rel_lm[key] = np.logaddexp(rel_lm[key], val + q_prob_in_doc)
  return defaultdict(lambda: -np.inf, {term: np.logaddexp(rel_lm[term] + np.log(smooth),
                                                          query_lm[term] + np.log(1 - smooth))
                                             for term in rel_lm})

def _calc_score_under_lm(lm, doc_f):
  score = -np.inf
  for term, cnt in doc_f.items():
    score = np.logaddexp(score, np.log(cnt) + lm[term])
  return score

def calc_docs_lms(corpus_fs, docs_fs, prior=2000):
  corpus_size = sum(corpus_fs.values())
  docs_lms = []
  for doc_fs in docs_fs:
    doc_lm = defaultdict(lambda: -np.inf)
    doc_len = sum(doc_fs.values())
    for term in doc_fs:
      doc_lm[term] = np.log((doc_fs[term] + corpus_fs[term] * prior / corpus_size) / doc_len)
    docs_lms.append(doc_lm)
  return docs_lms

def top_k(score_fn, doc_ids, k=10):
  score_pairs = [(score_fn(doc_id), doc_id) for doc_id in doc_ids]
  topk = nlargest(k,
                  score_pairs,
                  key=itemgetter(0))
  return [doc_id for score, doc_id in sorted(topk, key=itemgetter(0))]

def rank_rm3(docs_lms, docs_fs, qml_ranking, q, k=10):
  rel_lm = _get_rel_lm(docs_lms, qml_ranking, q)
  return top_k(lambda doc_id: _calc_score_under_lm(rel_lm, docs_fs[doc_id]),
               range(len(docs_lms)),
               k=k)

def get_other_results(queries, qml_rankings, num_ranks=None):
  document_lookup = read_cache('./doc_lookup.json', get_robust_documents)
  document_title_to_id = read_cache('./document_title_to_id.json',
                                    lambda: print('failed'))
  document_id_to_title = _.invert(document_title_to_id)
  doc_ids = range(len(document_id_to_title))
  documents = [document_lookup[document_id_to_title[doc_id]] for doc_id in doc_ids]
  tokenizer = Tokenizer(rules=[handle_caps, fix_html, spec_add_spaces, rm_useless_spaces])
  tokenized_documents = read_cache('tok_docs.json',
                                   lambda: tokenizer.process_all(documents))
  tokenized_queries = tokenizer.process_all(queries)
  bm25 = BM25(tokenized_documents)
  average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)
  bm25_rankings = []
  glove_rankings = []
  rm3_rankings = []
  glove = get_glove_lookup(embedding_dim=300, use_large_embed=True)
  docs_lms = calc_docs_lms(bm25.df, bm25.f)
  encoded_docs = torch.stack([_encode_glove_fs(glove, doc_fs) for doc_fs in docs_fs])
  encoded_docs = encoded_docs / torch.norm(encoded_docs, dim=1)
  for q, qml_ranking in progressbar(zip(tokenized_queries, qml_rankings)):
    bm25_rankings.append(rank_bm25(bm25, q, average_idf=average_idf))
    glove_rankings.append(rank_glove(glove, encoded_docs, q))
    rm3_rankings.append(rank_rm3(docs_lms, bm25.f, qml_ranking, q))
  return bm25_rankings, glove_rankings, rm3_rankings

def main():
  with open('./caches/pairwise_train_ranking_106756.json') as fh:
    query_ranking_pairs = json.load(fh)
  queries_by_tok_id, qml = zip(*query_ranking_pairs)
  parsed_queries, query_token_lookup = read_cache('./parsed_robust_queries_dict.json',
                                                  lambda: print('failed'))
  inv = _.invert(query_token_lookup)
  queries = [' '.join([inv[q] for q in query]) for query in queries_by_tok_id]
  if len(sys.argv) > 1:
    lim = int(sys.argv[1])
  else:
    lim = None
  bm25_rankings, glove_rankings, rm3_rankings = get_other_results(queries[:lim], qml[:lim])
  agree_ctr, num_combos = check_overlap(qml[:lim], bm25_rankings)
  print(agree_ctr, num_combos, agree_ctr/num_combos)
  agree_ctr, num_combos = check_overlap(qml[:lim], glove_rankings)
  print(agree_ctr, num_combos, agree_ctr/num_combos)
  agree_ctr, num_combos = check_overlap(qml[:lim], rm3_rankings)
  print(agree_ctr, num_combos, agree_ctr/num_combos)
