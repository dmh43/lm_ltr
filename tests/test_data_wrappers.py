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
