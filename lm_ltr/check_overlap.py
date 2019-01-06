from six.moves import xrange
import pydash as _
from itertools import combinations
import json

from gensim.summarization.bm25 import BM25
from fastai.text import Tokenizer
import numpy as np

from lm_ltr.fetchers import read_cache, get_robust_test_queries, get_robust_rels, get_robust_documents
from lm_ltr.preprocessing import create_id_lookup


def get_bm25_results(queries, num_ranks=None):
  def _get_scores(bm25, document, average_idf):
    scores = []
    for index in xrange(bm25.corpus_size):
      score = bm25.get_score(document, index, average_idf)
      scores.append(score)
    scores = np.array(scores)
    return scores[np.argpartition(scores, -10)[-10:]]
  document_lookup = read_cache('./doc_lookup.pkl', get_robust_documents)
  document_title_to_id = create_id_lookup(document_lookup.keys())
  document_id_to_title = _.invert(document_title_to_id)
  doc_ids = range(len(document_id_to_title))
  documents = [document_lookup[document_id_to_title[doc_id]] for doc_id in doc_ids]
  tokenizer = Tokenizer()
  tokenized_documents = read_cache('tok_docs.pkl',
                                   lambda: tokenizer.process_all(documents))
  tokenized_queries = read_cache('tok_queries.pkl',
                                 lambda: tokenizer.process_all(queries))
  bm25 = BM25(tokenized_documents)
  average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)
  return list(map(lambda document: _get_scores(bm25, document, average_idf=average_idf),
                  tokenized_queries))

def get_qml_results(queries, num_ranks=None):
  with open('./pairwise_train_ranking_106756.json') as fh:
    rankings = json.load(fh)
    return [rankings[str(query)[1:-1]] for query in queries]

def check_overlap(ranks_1, ranks_2):
  agree_ctr = 0
  num_combos = 0
  for doc_1, doc_2 in combinations(ranks_1, 2):
    num_combos += 1
    d_1_in_2 = _.index_of(ranks_2, doc_1)
    d_2_in_2 = _.index_of(ranks_2, doc_2)
    d_1_in_1 = _.index_of(ranks_1, doc_1)
    d_2_in_1 = _.index_of(ranks_1, doc_2)
    if any(rank == -1 for rank in [d_2_in_2, d_2_in_2, d_1_in_1, d_1_in_2]): continue
    if (d_1_in_1 < d_2_in_1) == (d_1_in_2 < d_2_in_2): agree_ctr += 1
  return agree_ctr, num_combos

def main():
  query_lookup = read_cache('./robust_test_queries.pkl', get_robust_test_queries)
  query_name_to_id = create_id_lookup(query_lookup.keys())
  query_id_to_name = _.invert(query_name_to_id)
  query_ids = range(len(query_id_to_name))
  queries = [query_lookup[query_id_to_name[query_id]] for query_id in query_ids]
  bm25 = get_bm25_results(queries)
  qml = get_qml_results(queries)
  agree_ctr, num_combos = check_overlap(bm25, qml)
  print(agree_ctr, num_combos, agree_ctr/num_combos)

if __name__ == "__main__": main()
