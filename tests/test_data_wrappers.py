import lm_ltr.data_wrappers as df

import scipy.sparse as sp
import torch

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
  assert df._get_nth_pair(rankings, 0) == {'query': [1],
                                           'doc_id_1': 3,
                                           'doc_id_2': 4,
                                           'rel': 1}
  assert df._get_nth_pair(rankings, 1) == {'query': [1],
                                           'doc_id_1': 3,
                                           'doc_id_2': 1,
                                           'rel': 1}
  assert df._get_nth_pair(rankings, 2) == {'query': [1],
                                           'doc_id_1': 4,
                                           'doc_id_2': 3,
                                           'rel': -1}
  assert df._get_nth_pair(rankings, 3) == {'query': [1],
                                           'doc_id_1': 4,
                                           'doc_id_2': 1,
                                           'rel': 1}
  assert df._get_nth_pair(rankings, 4) == {'query': [1],
                                           'doc_id_1': 1,
                                           'doc_id_2': 3,
                                           'rel': -1}
  assert df._get_nth_pair(rankings, 5) == {'query': [1],
                                           'doc_id_1': 1,
                                           'doc_id_2': 4,
                                           'rel': -1}
  assert df._get_nth_pair(rankings, 6) == {'query': [2, 3],
                                           'doc_id_1': 8,
                                           'doc_id_2': 9,
                                           'rel': 1}
