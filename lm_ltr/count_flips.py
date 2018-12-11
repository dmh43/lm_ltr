import numpy as np
import pydash as _

import sys

from lm_ltr.trec_doc_parse import parse_qrels
from lm_ltr.metrics import metrics_at_k
from lm_ltr.utils import append_at
from lm_ltr.fetchers import read_query_test_rankings

def main():
  rankings_to_eval = read_query_test_rankings()
  qrels = parse_qrels()
  queries = list(set(qrels.keys()).intersection(set(rankings_to_eval.keys())))
  ordered_qrels = [qrels[query] for query in queries]
  ordered_rankings_to_eval = [rankings_to_eval[query] for query in queries]
  k = 10 if len(sys.argv) == 1 else int(sys.argv[1])
  num_swaps = []
  for indri_rank, true_rank in zip(ordered_rankings_to_eval, ordered_qrels):
    true_docs = set(true_rank)
    irrel_ctr = 0
    swap_ctr = 0
    for pos, doc in enumerate(indri_rank):
      if doc in true_docs:
        swap_ctr += irrel_ctr
      else:
        irrel_ctr += 1
    num_swaps.append(swap_ctr)
  print(np.sum(num_swaps), np.mean(num_swaps), np.std(num_swaps))
  print(num_swaps)

if __name__ == "__main__":
  import ipdb
  import traceback
  import sys

  try:
    main()
  except: # pylint: disable=bare-except
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
  ipdb.post_mortem(tb)
