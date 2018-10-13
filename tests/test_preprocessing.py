import torch
import lm_ltr.preprocessing as p

def test_to_ranking_per_query():
  data = [{'query': [1, 2],
           'doc_id': i*10,
           'rank': i} for i in range(10)]
  assert p.to_query_rankings_pairs(data) == [[[1, 2], list(range(0, 100, 10))]]

def test_to_ranking_per_query__multiple():
  data = [{'query': [1, 2],
           'doc_id': i*10,
           'rank': i} for i in range(10)] + [{'query': [3],
                                              'doc_id': i*10,
                                              'rank': i} for i in range(10)]
  assert p.to_query_rankings_pairs(data) == [[[1, 2], list(range(0, 100, 10))],
                                             [[3], list(range(0, 100, 10))]]
