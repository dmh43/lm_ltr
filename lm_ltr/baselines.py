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

def _encode_glove(glove, idf, tokens):
  weights = torch.tensor([idf[token] if token in idf else 0.0 for token in tokens]).float().cuda()
  tok_vecs = torch.stack([glove[token] for token in tokens if token in glove]).cuda()
  weighted_tokens = weights.unsqueeze(1) * tok_vecs
  vec = torch.sum(weighted_tokens, 0)
  return vec / torch.norm(vec)

def encode_glove_fs(glove, idf, doc_fs):
  weights = torch.tensor([idf[token] if token in idf else 0.0 for token in doc_fs]).float().cuda()
  words = torch.stack([glove[token]
                       for token, cnt in doc_fs.items() if token in glove and cnt > 0]).cuda()
  freqs = torch.tensor([cnt
                        for token, cnt in doc_fs.items() if token in glove and cnt > 0], dtype=torch.float).cuda()
  return torch.sum(words * freqs.unsqueeze(1) * weights.unsqueeze(1), 0)

def rank_glove(glove, idf, encoded_docs, query, k=10):
  topk_scores, topk_idxs = torch.topk(torch.sum(encoded_docs * _encode_glove(glove, idf, query), 1),
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

def _calc_score_under_lm(lm, doc_lm):
  score = 0
  for term, log_prob in doc_lm.items():
    score += np.exp(log_prob) * lm[term]
  return score

def calc_docs_lms(corpus_fs, docs_fs, prior=2000):
  corpus_size = sum(corpus_fs.values())
  docs_lms = []
  for doc_fs in docs_fs:
    doc_lm = defaultdict(lambda: -np.inf)
    doc_len = sum(doc_fs.values())
    for term in doc_fs:
      doc_lm[term] = np.log((doc_fs[term] + corpus_fs[term] * prior / corpus_size) / (doc_len + prior))
    docs_lms.append(doc_lm)
  return docs_lms

def top_k(score_fn, doc_ids, k=10):
  score_pairs = [(score_fn(doc_id), doc_id) for doc_id in doc_ids]
  topk = nlargest(k,
                  score_pairs,
                  key=itemgetter(0))
  return [doc_id for score, doc_id in sorted(topk, key=itemgetter(0))]

def rank_rm3(docs_lms, qml_ranking, q, k=10):
  rel_lm = _get_rel_lm(docs_lms, qml_ranking, q)
  return top_k(lambda doc_id: _calc_score_under_lm(rel_lm, docs_lms[doc_id]),
               range(len(docs_lms)),
               k=k)
