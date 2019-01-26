import pydash as _
import numpy as np

import lm_ltr.snorkel_helper as r

def test_get_pairwise_bins():
  pos, neg = r.get_pairwise_bins([1, 2, 3, 4])
  assert len(pos) == 6
  assert len(neg) == 6
  assert len(pos.union(neg)) == 12

def test_L_from_rankings_shape():
  _triangle = lambda num: (num * (num + 1)) / 2 - num
  rankings = {'tfidf' : [[2, 1, 3, 4],
                         [1, 0, 2, 3]],
              'bm25'  : [[1, 2, 0, 3],
                         [1, 2, 3, 4]],
              'qml'   : [[2, 5, 7, 1],
                         [2, 5, 3, 1]],
              'fb'    : [[2, 5, 3, 1],
                         [2, 5, 7, 1]]}
  L = r.get_L_from_rankings(rankings).toarray()
  assert L.shape[0] == sum(_triangle(num_uniq) for num_uniq in [7, 7])
  assert L.shape[1] == 4

def test_L_from_rankings_vals():
  rankings = {'tfidf' : [[2, 1]],
              'bm25'  : [[1, 2]]}
  L = r.get_L_from_rankings(rankings).toarray()
  assert L.shape[0] == 1
  assert L.shape[1] == 2
  assert np.array_equal(L, np.array([[1, -1]])) or np.array_equal(L, np.array([[-1, 1]]))

def test_L_from_rankings_vals_abstain():
  rankings = {'tfidf' : [[2, 1]],
              'bm25'  : [[1, 2]],
              'qml'   : [[2, 5]]}
  L = r.get_L_from_rankings(rankings).toarray()
  assert L.shape[0] == 3
  assert L.shape[1] == 3
  row_options = [[1, -1, 0],
                 [0, 0, 1],
                 [-1, 1, 0],
                 [0, 0, -1],
                 [0, 0, 0]]
  assert sum(any(row == row_option for row in L.tolist())
             for row_option in row_options) == 3

def test_L_from_pairs_vals():
  target_infos = [((2, 1), [98, 208, 0])]
  query_pairwise_bins_by_ranker = {'tfidf' : {'98, 208, 0': ({(1, 2)}, {(2, 1)})},
                                   'bm25'  : {'98, 208, 0': ({(2, 1)}, {(1, 2)})}}
  L = r.get_L_from_pairs(query_pairwise_bins_by_ranker, target_infos).toarray()
  assert L.shape[0] == 1
  assert L.shape[1] == 2
  assert np.array_equal(L, np.array([[1, -1]])) or np.array_equal(L, np.array([[-1, 1]]))

def test_L_from_pairs_more_vals():
  target_infos = [((2, 1), [98, 208, 0]),
                  ((10, 3), [98, 208, 0]),
                  ((5, 7), [0, 0, 0])]
  query_pairwise_bins_by_ranker = {'tfidf' : {'98, 208, 0' : ({(1, 2), (10, 3)}, {(2, 1), (3, 10)}),
                                              '0, 0, 0'    : ({(5, 7)}, {(7, 5)})},
                                   'bm25'  : {'98, 208, 0' : ({(2, 1), (10, 3)}, {(1, 2), (3, 10)}),
                                              '0, 0, 0'    : ({(5, 10)}, {(10, 5)})}}
  L = r.get_L_from_pairs(query_pairwise_bins_by_ranker, target_infos).toarray()
  assert False
  assert L.shape[0] == 3
  assert L.shape[1] == 2
  assert  np.array_equal(L, np.array([[-1, 1], [1, 1], [1, 0]]))
