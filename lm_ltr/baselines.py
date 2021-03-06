from six.moves import xrange
import pydash as _
from itertools import combinations
import json
from collections import Counter, defaultdict
from operator import itemgetter
from heapq import nlargest
from collections.abc import MutableMapping

from gensim.summarization.bm25 import BM25
from fastai.text import Tokenizer, fix_html, spec_add_spaces, rm_useless_spaces
import numpy as np
from progressbar import progressbar
import torch

from lm_ltr.fetchers import read_cache, get_robust_eval_queries, get_robust_rels, get_robust_documents
from lm_ltr.preprocessing import create_id_lookup, handle_caps
from lm_ltr.embedding_loaders import get_glove_lookup


def rank_bm25(bm25, q, average_idf, doc_ids=None):
  doc_ids = range(bm25.corpus_size) if doc_ids is None else doc_ids
  return top_k(lambda doc_id: bm25.get_score(q, doc_id, average_idf),
               doc_ids,
               k=10)

def _encode_glove(glove, idf, tokens, default=1.0):
  w = []
  for token in tokens:
    if token in glove:
      if token in idf:
        w.append(idf[token])
  weights = torch.tensor(w).float().cuda()
  tok_vecs = torch.stack([glove[token] for token in tokens if token in glove]).cuda()
  weighted_tokens = weights.unsqueeze(1) * tok_vecs
  vec = torch.sum(weighted_tokens, 0)
  return vec / torch.norm(vec)

def encode_glove_fs(glove, idf, doc_fs):
  weights = torch.tensor([idf[token]
                          for token, cnt in doc_fs.items() if token in glove and cnt > 0]).float().cuda()
  words = torch.stack([glove[token]
                       for token, cnt in doc_fs.items() if token in glove and cnt > 0]).cuda()
  freqs = torch.tensor([cnt
                        for token, cnt in doc_fs.items() if token in glove and cnt > 0], dtype=torch.float).cuda()
  return torch.sum(words * freqs.unsqueeze(1) * weights.unsqueeze(1), 0)

def rank_glove(glove, idf, encoded_docs, query, k=10, doc_ids=None):
  doc_ids = torch.arange(encoded_docs.shape[0], device=encoded_docs.device) if doc_ids is None else doc_ids
  avg = sum(idf.values()) / len(idf)
  topk_scores, topk_idxs = torch.topk(torch.sum(encoded_docs[doc_ids] * _encode_glove(glove, idf, query, default=avg), 1),
                                      k=min(k, len(doc_ids)))
  sorted_scores, sort_idxs = torch.sort(topk_scores, descending=True)
  return torch.tensor(doc_ids).cuda()[topk_idxs[sort_idxs]].tolist()


class LM(MutableMapping):
  def __init__(self, fs, corpus_fs, prior, corpus_size):
    self.fs = fs
    self.prior = prior
    self.corpus_size = corpus_size
    self.corpus_fs = corpus_fs
    self.doc_len = sum(fs.values())
    self.store = {}
    self.default = np.log(self.prior / self.corpus_size / (self.doc_len + self.prior))

  def __getitem__(self, key):
    if key in self.store:
      return self.store[key]
    else:
      if key in self.corpus_fs:
        return np.log(self.corpus_fs[key]) + self.default
      else:
        return self.default

  def __setitem__(self, key, val):
    self.store[key] = val

  def __delitem__(self, key):
    del self.store[key]

  def __repr__(self):
    return repr(self.store)

  def update(self, *args, **kwargs):
    for k, v in dict(*args, **kwargs).items():
      self.store[k] = v

  def __iter__(self):
    return iter(self.store)

  def __len__(self):
    return len(self.store)


def calc_docs_lms(corpus_fs, docs_fs, prior=2000):
  corpus_size = sum(corpus_fs.values())
  docs_lms = []
  for doc_fs in docs_fs:
    doc_len = sum(doc_fs.values())
    doc_lm = LM(doc_fs, corpus_fs, prior, corpus_size)
    for term in doc_fs:
      doc_lm[term] = np.log((doc_fs[term] + corpus_fs[term] * prior / corpus_size) / (doc_len + prior))
    docs_lms.append(doc_lm)
  return docs_lms

def _get_rel_lm(docs_lms, qml_ranking, q, smooth=0.5):
  query_lm = defaultdict(lambda: -np.inf,
                         {q_term: np.log(cnt / len(q)) for q_term, cnt in Counter(q).items()})
  rel_lm = defaultdict(lambda: -np.inf)
  for doc_id in qml_ranking:
    q_prob_in_doc = np.sum([docs_lms[doc_id][q_term] for q_term in q])
    for term in docs_lms[doc_id]:
      rel_lm[term] = np.logaddexp(rel_lm[term], docs_lms[doc_id][term] + q_prob_in_doc)
  return defaultdict(lambda: -np.inf, {term: np.logaddexp(rel_lm[term] + np.log(smooth),
                                                          query_lm[term] + np.log(1 - smooth))
                                             for term in rel_lm})

def _get_top_n_terms(lm, n=30):
  return _.map_(nlargest(n, _.to_pairs(lm), itemgetter(1)),
                itemgetter(0))

def _calc_score_under_lm(lm, doc_lm, top_n_terms):
  exp = np.exp(np.array([lm[term] for term in top_n_terms]))
  probs = np.array([doc_lm[term] for term in top_n_terms])
  return np.sum(exp * probs)

def rank_rm3(docs_lms, qml_ranking, q, k=10, doc_ids=None, smooth=0.5):
  doc_ids = range(len(docs_lms)) if doc_ids is None else doc_ids
  rel_lm = _get_rel_lm(docs_lms, qml_ranking, q, smooth=smooth)
  top_n_terms = _get_top_n_terms(rel_lm)
  return top_k(lambda doc_id: _calc_score_under_lm(rel_lm, docs_lms[doc_id], top_n_terms),
               doc_ids,
               k=k)

def rank_lsi(lsi, tfidf, q_numericalized, doc_ids=None):
  doc_ids = range(lsi.docs_processed) if doc_ids is None else doc_ids
  tfidf[q_numericalized]

def top_k(score_fn, doc_ids, k=10):
  score_pairs = [(score_fn(doc_id), doc_id) for doc_id in doc_ids]
  topk = nlargest(k,
                  score_pairs,
                  key=itemgetter(0))
  return [doc_id for score, doc_id in sorted(topk, key=itemgetter(0))]
