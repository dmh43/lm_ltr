from six.moves import xrange
import pydash as _
from itertools import combinations
import json

from gensim.summarization.bm25 import BM25
from fastai.text import Tokenizer
import numpy as np
from progressbar import progressbar

from lm_ltr.fetchers import read_cache, get_robust_test_queries, get_robust_rels, get_robust_documents
from lm_ltr.preprocessing import create_id_lookup


def get_bm25_results(queries, qml_rankings, num_ranks=None):
  def _get_scores(bm25, qml_ranking, document, average_idf):
    scores = []
    for doc_id in qml_ranking:
      score = bm25.get_score(document, doc_id, average_idf)
      scores.append(score)
    scores = np.array(scores)
    return [qml_ranking[idx] for idx in np.argsort(-scores)]
  document_lookup = read_cache('./doc_lookup.json', get_robust_documents)
  document_title_to_id = read_cache('./document_title_to_id.json',
                                    lambda: print('failed'))
  document_id_to_title = _.invert(document_title_to_id)
  doc_ids = range(len(document_id_to_title))
  documents = [document_lookup[document_id_to_title[doc_id]] for doc_id in doc_ids]
  tokenizer = Tokenizer()
  tokenized_documents = read_cache('tok_docs.json',
                                   lambda: tokenizer.process_all(documents))
  tokenized_queries = tokenizer.process_all(queries)
  bm25 = BM25(tokenized_documents)
  average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)
  rankings = []
  for q, qml_ranking in progressbar(zip(tokenized_queries, qml_rankings)):
    rankings.append(_get_scores(bm25, qml_ranking, q, average_idf=average_idf))
  return rankings

def check_overlap(ranks_1, ranks_2):
  agree_ctr = 0
  num_combos = 0
  for ranks_1, ranks_2 in zip(ranks_1, ranks_2):
    for doc_1, doc_2 in combinations(ranks_1, 2):
      num_combos += 1
      d_1_in_2 = _.index_of(ranks_2, doc_1)
      d_2_in_2 = _.index_of(ranks_2, doc_2)
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
  bm25 = get_bm25_results(queries[:lim], qml[:lim])
  agree_ctr, num_combos = check_overlap(bm25, qml[:lim])
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
