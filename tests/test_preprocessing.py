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

def test_normalize_scores_query_wise():
  data = [{'query': [1], 'doc_id': 1, 'score': -11.0},
          {'query': [1], 'doc_id': 2, 'score': -11.3},
          {'query': [1], 'doc_id': 3, 'score': -11.9},
          {'query': [1], 'doc_id': 4, 'score': -12.0},
          {'query': [1], 'doc_id': 5, 'score': -13.0},
          {'query': [1, 2], 'doc_id': 1, 'score': -1.0},
          {'query': [1, 2], 'doc_id': 2, 'score': -1.3},
          {'query': [1, 2], 'doc_id': 3, 'score': -1.9},
          {'query': [1, 2], 'doc_id': 4, 'score': -2.0},
          {'query': [1, 2], 'doc_id': 5, 'score': -3.0}]
  normalized = p.normalize_scores_query_wise(data)
  assert len(normalized) == len(data)
  assert abs(sum([torch.exp(torch.tensor(row['score']))
                  for row in normalized[:5]]) - torch.tensor(1.0)) < 1e-6
  assert abs(sum([torch.exp(torch.tensor(row['score']))
                  for row in normalized[5:]]) - torch.tensor(1.0)) < 1e-6
