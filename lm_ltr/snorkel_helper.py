from typing import List, Dict, Set, Tuple
from functools import reduce
from itertools import combinations

from scipy.sparse import csr_matrix
import numpy as np

def get_all_items(ranked_lists: List[List[int]]):
  return list(reduce(lambda acc, ranking: acc.union(ranking), ranked_lists, set()))

def get_pairwise_bins(ranking):
  pos, neg = set(), set()
  for a, b in combinations(ranking, 2):
    pos.add((a, b))
    neg.add((b, a))
  return pos, neg

def get_ranker_output(pairwise_bins, pair):
  pos, neg = pairwise_bins
  if pair in pos: return 1
  elif pair in neg: return -1
  else: return 0

def get_L_from_rankings(all_ranked_lists_by_ranker: Dict[str, List[List[int]]]) -> csr_matrix :
  rankings_per_query = zip(*all_ranked_lists_by_ranker.values())
  num_lfs = len(all_ranked_lists_by_ranker)
  num_rows = 0
  pair_idx = 0
  data = []
  row_ind = []
  col_ind = []
  for ranked_lists in rankings_per_query:
    all_items = get_all_items(ranked_lists)
    rankings_bins = [get_pairwise_bins(ranking) for ranking in ranked_lists]
    for pair in combinations(all_items, 2):
      for lf_idx, pairwise_bins in enumerate(rankings_bins):
        print(lf_idx)
        output = get_ranker_output(pairwise_bins, pair)
        if output != 0:
          data.append(output)
          row_ind.append(pair_idx)
          col_ind.append(lf_idx)
          num_rows = max(num_rows, pair_idx + 1)
      pair_idx += 1
  return csr_matrix((data, (row_ind, col_ind)), shape=(num_rows, num_lfs))

def get_L_from_pairs(query_pairwise_bins_by_ranker: Dict[str, Dict[str, Tuple[Set[Tuple[int, int]],
                                                                              Set[Tuple[int, int]]]]],
                     target_infos: List[Tuple[Tuple[int, int], List[int]]]) -> csr_matrix:
  num_lfs = len(query_pairwise_bins_by_ranker)
  num_rows = 0
  pair_idx = 0
  data = []
  row_ind = []
  col_ind = []
  for target_info in target_infos:
    for lf_idx, ranker_pairwise_bins_by_query in enumerate(query_pairwise_bins_by_ranker.values()):
      ranker_pairwise_bins = ranker_pairwise_bins_by_query[str(target_info[1])[1:-1]]
      pair = target_info[0]
      output = get_ranker_output(ranker_pairwise_bins, pair)
      if output != 0:
        data.append(output)
        row_ind.append(pair_idx)
        col_ind.append(lf_idx)
        num_rows = max(num_rows, pair_idx + 1)
    pair_idx += 1
  return csr_matrix((data, (row_ind, col_ind)), shape=(num_rows, num_lfs))