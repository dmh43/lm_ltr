import lm_ltr.data_wrappers as df

import scipy.sparse as sp
import torch

def test__get_top_scoring():
  tfidf_docs = sp.lil_matrix((10, 10))
  tfidf_docs[0, 1] = 10
  tfidf_docs[1, 3] = 30
  tfidf_docs[4, 3] = 20
  tfidf_docs[5, 8] = 1000
  query = torch.tensor([1, 3], dtype=torch.long)
  result = df._get_top_scoring(tfidf_docs, query, k=2).tolist()
  assert 1 in result
  assert 4 in result
  assert 5 not in result
