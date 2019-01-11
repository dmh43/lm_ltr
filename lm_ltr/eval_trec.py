import pydash as _

import sys
from gensim.summarization.bm25 import BM25
from fastai.text import Tokenizer, fix_html, spec_add_spaces, rm_useless_spaces
from progressbar import progressbar

from lm_ltr.trec_doc_parse import parse_qrels
from lm_ltr.metrics import metrics_at_k
from lm_ltr.utils import append_at
from lm_ltr.fetchers import read_query_test_rankings, read_cache, get_robust_test_queries, get_robust_rels, get_robust_documents
from lm_ltr.preprocessing import create_id_lookup, handle_caps
from lm_ltr.embedding_loaders import get_glove_lookup
from lm_ltr.baselines import calc_docs_lms, rank_rm3, rank_glove, rank_bm25

def main():
  rankings_to_eval = read_query_test_rankings()
  qrels = parse_qrels()
  queries = list(set(qrels.keys()).intersection(set(rankings_to_eval.keys())))
  ordered_qrels = [qrels[query] for query in queries]
  ordered_rankings_to_eval = [rankings_to_eval[query] for query in queries]
  k = 10 if len(sys.argv) == 1 else int(sys.argv[1])
  document_lookup = read_cache('./doc_lookup.json', get_robust_documents)
  document_title_to_id = read_cache('./document_title_to_id.json',
                                    lambda: print('failed'))
  document_id_to_title = _.invert(document_title_to_id)
  doc_ids = range(len(document_id_to_title))
  documents = [document_lookup[document_id_to_title[doc_id]] for doc_id in doc_ids]
  tokenizer = Tokenizer(rules=[handle_caps, fix_html, spec_add_spaces, rm_useless_spaces])
  tokenized_documents = read_cache('tok_docs.json',
                                   lambda: tokenizer.process_all(documents))
  tokenized_queries = tokenizer.process_all(queries)
  bm25 = BM25(tokenized_documents)
  average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)
  bm25_rankings = []
  glove_rankings = []
  rm3_rankings = []
  glove = get_glove_lookup(embedding_dim=300, use_large_embed=True)
  docs_lms = calc_docs_lms(bm25.df, bm25.f)
  for q, qml_ranking in progressbar(zip(tokenized_queries, ordered_rankings_to_eval),
                                    max_value=len(tokenized_queries)):
    bm25_rankings.append(rank_bm25(bm25, q, average_idf=average_idf))
    glove_rankings.append(rank_glove(glove, bm25.f, q))
    rm3_rankings.append(rank_rm3(docs_lms, bm25.f, qml_ranking, q))
  print('indri:', metrics_at_k(ordered_rankings_to_eval, ordered_qrels, k))
  print('bm25:', metrics_at_k(bm25_rankings, ordered_qrels, k))
  print('glove:', metrics_at_k(glove_rankings, ordered_qrels, k))
  print('rm3:', metrics_at_k(rm3_rankings, ordered_qrels, k))

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
