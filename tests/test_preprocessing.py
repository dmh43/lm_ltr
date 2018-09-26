import torch
import lm_ltr.preprocessing as p

def test_to_ranking_per_query():
  data = [{'query': [1, 2],
           'document_id': i*10,
           'rank': i} for i in range(10)]
  assert p.to_query_rankings_pairs(data) == [[[1, 2], list(range(0, 100, 10))]]

def test_to_ranking_per_query_at_k():
  data = [{'query': [1, 2],
           'document_id': i*10,
           'rank': i} for i in range(10)]
  assert p.to_query_rankings_pairs(data, k=4) == [[[1, 2], list(range(0, 40, 10))]]
