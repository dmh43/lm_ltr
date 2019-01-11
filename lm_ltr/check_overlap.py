from six.moves import xrange
import pydash as _
from itertools import combinations
import json
from collections import Counter, defaultdict

from gensim.summarization.bm25 import BM25
from fastai.text import Tokenizer, fix_html, spec_add_spaces, rm_useless_spaces
import numpy as np
from progressbar import progressbar
import torch

from lm_ltr.fetchers import read_cache, get_robust_test_queries, get_robust_rels, get_robust_documents
from lm_ltr.preprocessing import create_id_lookup, handle_caps
from lm_ltr.embedding_loaders import get_glove_lookup


def _get_bm25_ranking(bm25, qml_ranking, document, average_idf):
  if len(sys.argv) > 2:
    qml_ranking = xrange(bm25.corpus_size)
  scores = []
  for doc_id in qml_ranking:
    score = bm25.get_score(document, doc_id, average_idf)
    scores.append(score)
  scores = np.array(scores)
  return [qml_ranking[idx] for idx in np.argsort(-scores)]

def _encode_glove(glove, tokens):
  vec = torch.sum(torch.stack([glove[token] for token in tokens if token in glove]).cuda(), 0)
  return vec / torch.norm(vec)

def _get_glove_ranking(glove, documents, qml_ranking, query):
  if len(sys.argv) > 2:
    qml_ranking = xrange(len(documents))
  encoded_docs = torch.stack([_encode_glove(glove, documents[doc_id]) for doc_id in qml_ranking])
  return [qml_ranking[idx]
          for idx in torch.sort(torch.sum(encoded_docs * _encode_glove(glove, query), 1),
                                descending=True)[1]]

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

def _calc_docs_lms(corpus_fs, docs_fs, prior=2000):
  corpus_size = sum(corpus_fs.values())
  docs_lms = []
  for doc_fs in docs_fs:
    doc_lm = defaultdict(lambda: -np.inf)
    doc_len = sum(doc_fs.values())
    for term in doc_fs:
      doc_lm[term] = np.log((doc_fs[term] + corpus_fs[term] * prior / corpus_size) / doc_len)
    docs_lms.append(doc_lm)
  return docs_lms

def _get_rm3_ranking(docs_lms, docs_fs, qml_ranking, q):
  rel_lm = _get_rel_lm(docs_lms, qml_ranking, q)
  return [qml_ranking[idx]
          for idx in np.argsort([_calc_score_under_lm(rel_lm, docs_fs[doc_id])
                                 for doc_id in qml_ranking])]

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
  docs_lms = _calc_docs_lms(bm25.df, bm25.f)
  for q, qml_ranking in progressbar(zip(tokenized_queries, qml_rankings)):
    bm25_rankings.append(_get_bm25_ranking(bm25, qml_ranking, q, average_idf=average_idf))
    glove_rankings.append(_get_glove_ranking(glove, tokenized_documents, qml_ranking, q))
    rm3_rankings.append(_get_rm3_ranking(docs_lms, bm25.f, qml_ranking, q))
  return bm25_rankings, glove_rankings, rm3_rankings

def check_overlap(ranks_1, ranks_2):
  agree_ctr = 0
  num_combos = 0
  for ranks_1, ranks_2 in zip(ranks_1, ranks_2):
    for doc_1, doc_2 in combinations(ranks_1, 2):
      num_combos += 1
      d_1_in_2 = _.index_of(ranks_2[:len(ranks_1)], doc_1)
      d_2_in_2 = _.index_of(ranks_2[:len(ranks_1)], doc_2)
      d_1_in_1 = _.index_of(ranks_1, doc_1)
      d_2_in_1 = _.index_of(ranks_1, doc_2)
      if d_1_in_2 == -1: continue
      if d_2_in_2 == -1:
        agree_ctr += 1
        continue
      if (d_1_in_1 < d_2_in_1) == (d_1_in_2 < d_2_in_2):
        agree_ctr += 1
  return agree_ctr, num_combos

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
