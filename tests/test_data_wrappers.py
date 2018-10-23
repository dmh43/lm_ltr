import lm_ltr.data_wrappers as df
import pydash as _

import scipy.sparse as sp
import torch
import numpy as np

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
  normalized = df.normalize_scores_query_wise(data)
  assert len(normalized) == len(data)
  assert abs(sum([torch.exp(row['score']) for row in normalized[:5]]) - torch.tensor(1.0)) < 1e-6
  assert abs(sum([torch.exp(row['score'])for row in normalized[5:]]) - torch.tensor(1.0)) < 1e-6
