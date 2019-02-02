from typing import List, Dict, Set, Tuple, Iterable
from functools import reduce
from itertools import combinations

from scipy.sparse import csr_matrix
import numpy as np
from snorkel.learning import GenerativeModel
from snorkel.learning.structure import DependencySelector
import pydash as _

from .types import TargetInfo, QueryPairwiseBinsByRanker, PairwiseBins

def get_all_items(ranked_lists: Iterable[List[int]]):
  all_items: Set = reduce(lambda acc, ranking: acc.union(ranking), ranked_lists, set())
  return list(all_items)

def get_pairwise_bins(ranking) -> PairwiseBins:
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

def get_L_from_rankings(all_ranked_lists_by_ranker: Dict[str, List[List[int]]]) -> csr_matrix:
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
        output = get_ranker_output(pairwise_bins, pair)
        if output != 0:
          data.append(output)
          row_ind.append(pair_idx)
          col_ind.append(lf_idx)
          num_rows = max(num_rows, pair_idx + 1)
      pair_idx += 1
  L = csr_matrix((data, (row_ind, col_ind)), shape=(num_rows, num_lfs))
  return L[(L != 0).sum(1).squeeze().nonzero()[1]]

def get_L_from_pairs(query_pairwise_bins_by_ranker: QueryPairwiseBinsByRanker,
                     target_infos: List[TargetInfo]) -> csr_matrix:
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


class Snorkeller:
  def __init__(self, query_pairwise_bins_by_ranker: QueryPairwiseBinsByRanker):
    self.query_pairwise_bins_by_ranker = query_pairwise_bins_by_ranker
    self.snorkel_gm = GenerativeModel()
    self.is_trained = False

  def train(self, train_ranked_lists_by_ranker: Dict[str, List[List[int]]]):
    L_train = get_L_from_rankings(train_ranked_lists_by_ranker)
    ds = DependencySelector()
    deps = ds.select(L_train, threshold=0.0)
    self.snorkel_gm.train(L_train, deps, epochs=100, decay=0.95, step_size=0.1 / L_train.shape[0], reg_param=1e-6)
    self.is_trained = True

  def calc_marginals(self, target_info: List[TargetInfo]):
    non_rand_target_info: List[TargetInfo] = []
    deltas: List[int] = []
    delta_idxs: List[int] = []
    for idx, info in enumerate(target_info):
      if isinstance(info, int):
        deltas.append(deltas[-1] + 1 if len(deltas) != 0 else 0)
        delta_idxs.append(idx)
      else:
        non_rand_target_info.append(info)
    offset = len(non_rand_target_info)
    order = []
    marginal_ctr = 0
    for idx in range(len(target_info)):
      if len(delta_idxs) != 0 and idx == delta_idxs[0]:
        order.append(offset + deltas[0])
        deltas = deltas[1:]
        delta_idxs = delta_idxs[1:]
      else:
        order.append(marginal_ctr)
        marginal_ctr += 1
    order = np.array(order)
    L = get_L_from_pairs(self.query_pairwise_bins_by_ranker, non_rand_target_info)
    marginals = self.snorkel_gm.marginals(L)
    all_marginals = np.concatenate([marginals, np.ones(len(target_info) - len(marginals), dtype=marginals.dtype)])
    return all_marginals[order]
