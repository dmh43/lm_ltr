
from lm_ltr.trec_doc_parse import parse_qrels
from lm_ltr.metrics import metrics_at_k
from lm_ltr.utils import append_at

def read_query_test_rankings(path='./indri/query_result_test'):
  rankings = {}
  with open(path) as fh:
    for line in fh:
      query_name, __, doc_title, doc_rank, doc_score, ___ = line.strip().split(' ')
      append_at(rankings, query_name, doc_title)
    return rankings

def main():
  rankings_to_eval = read_query_test_rankings()
  qrels = parse_qrels()
  queries = list(set(qrels.keys()).intersection(set(rankings_to_eval.keys())))
  ordered_qrels = [qrels[query] for query in queries]
  ordered_rankings_to_eval = [rankings_to_eval[query] for query in queries]
  print(metrics_at_k(ordered_rankings_to_eval, ordered_qrels))

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