import lm_ltr.data_wrappers as df
import pydash as _

import scipy.sparse as sp
import torch
import numpy as np

def test_query_pairwise_dataset():
  rankings = [[[1], [1, 2, 3]],
              [[9], [3, 4, 5, 6]],
              [[9, 3], [0, 5, 2, 1]]]
  documents = [range(10) for doc in range(7)]
  dataset = df.QueryPairwiseDataset(documents, [], num_neg_samples=3, rankings=rankings)
  assert len(dataset) == (3 + 3) ** 2 - (3 + 3) + 2 * ((4 + 3) ** 2 - (4 + 3))

def test__get_nth_pair():
  rankings = [[[1], [3, 4, 1]],
              [[2, 3], [8, 9, 4]]]
  num_pairs_per_ranking = _.map_(rankings, lambda ranking: len(ranking[1]) ** 2 - len(ranking[1]))
  cumu_ranking_lengths = np.cumsum(num_pairs_per_ranking)
  assert df._get_num_pairs(rankings) == (3 ** 2 - 3) * 2
  for i in range(df._get_num_pairs(rankings)):
    pair = df._get_nth_pair(rankings, cumu_ranking_lengths, i)
    assert isinstance(pair['query'], list)
    assert isinstance(pair['target_info'], int)
    assert isinstance(pair['doc_id_1'], int)
    assert isinstance(pair['doc_id_2'], int)

def test_true_random_sampler():
  dataset = list(range(10))
  sampler = df.TrueRandomSampler(dataset)
  assert len(sampler) == 10
  counter = 0
  for elem in sampler:
    assert elem in dataset
    counter += 1
  assert counter == 10
