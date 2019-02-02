from typing import List, Dict, Set, Tuple, Union

PairwiseBin = Set[Tuple[int, int]]
PairwiseBins = Tuple[PairwiseBin, PairwiseBin]
QueryPairwiseBins = Dict[str, PairwiseBins]
QueryPairwiseBinsByRanker = Dict[str, QueryPairwiseBins]

DocIdPair = Tuple[int, int]
Query = List[int]
TargetInfo = Tuple[DocIdPair, Query, int]
