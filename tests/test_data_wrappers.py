import lm_ltr.data_wrappers as df
import pydash as _

import scipy.sparse as sp
import torch
import numpy as np

def test__get_top_scoring():
  tfidf_docs = sp.lil_matrix((10, 10))
  tfidf_docs[0, 1] = 10
  tfidf_docs[1, 3] = 30
  tfidf_docs[4, 3] = 20
  tfidf_docs[5, 8] = 1000
  query = [1, 3]
  query_document_token_mapping = {i:i for i in range(10)}
  result = df.get_top_k(df.score_documents_tfidf(query_document_token_mapping, tfidf_docs, query), k=2).tolist()
  assert 1 in result
  assert 4 in result
  assert 5 not in result

def test__get_nth_pair():
  rankings = [[[1], [3, 4, 1]],
              [[2, 3], [8, 9, 4]]]
  num_pairs_per_ranking = _.map_(rankings, lambda ranking: len(ranking[1]) ** 2 - len(ranking[1]))
  cumu_ranking_lengths = np.cumsum(num_pairs_per_ranking)
  assert df._get_num_pairs(rankings) == (3 ** 2 - 3) * 2
  for i in range(df._get_num_pairs(rankings)):
    pair = df._get_nth_pair(rankings, cumu_ranking_lengths, i)
    assert isinstance(pair['query'], list)
    assert isinstance(pair['order_int'], int)
    assert isinstance(pair['doc_id_1'], int)
    assert isinstance(pair['doc_id_2'], int)
