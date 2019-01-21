import pydash as _

import sys
from gensim.summarization.bm25 import BM25
from fastai.text import Tokenizer, fix_html, spec_add_spaces, rm_useless_spaces
from progressbar import progressbar
import torch

from lm_ltr.trec_doc_parse import parse_qrels
from lm_ltr.metrics import metrics_at_k
from lm_ltr.utils import append_at, name
from lm_ltr.fetchers import read_query_test_rankings, read_cache, get_robust_eval_queries, get_robust_rels, get_robust_documents_with_titles
from lm_ltr.preprocessing import create_id_lookup, handle_caps
from lm_ltr.embedding_loaders import get_glove_lookup
from lm_ltr.baselines import calc_docs_lms, rank_rm3, rank_glove, rank_bm25, encode_glove_fs

def basic_eval():
  path = './indri/query_result_test'
  path = path + '/' + sys.argv[3] if len(sys.argv) >= 4 else path
  rankings_to_eval = read_query_test_rankings(path=path)
  qrels = parse_qrels()
  query_ids = list(qrels.keys())
  k = 10 if len(sys.argv) == 1 else int(sys.argv[1])
  document_title_to_id = read_cache('./document_title_to_id.json',
                                    lambda: print('failed'))
  ordered_rankings_to_eval = [[document_title_to_id[title] for title in rankings_to_eval[query]]
                              for query in query_ids]
  ordered_qrels = [[document_title_to_id[title] for title in qrels[query]]
                   for query in query_ids]
  print('indri:', metrics_at_k(ordered_rankings_to_eval, ordered_qrels, k))

def baselines_eval():
  rankings_to_eval = read_query_test_rankings()
  qrels = parse_qrels()
  query_ids = list(qrels.keys())
  query_lookup = get_robust_eval_queries()
  queries = [query_lookup[query_id] for query_id in query_ids]
  k = 10 if len(sys.argv) == 1 else int(sys.argv[1])
  document_lookup = read_cache(name('./doc_lookup.json', ['with_titles']), get_robust_documents_with_titles)
  document_title_to_id = read_cache('./document_title_to_id.json',
                                    lambda: print('failed'))
  ordered_rankings_to_eval = [[document_title_to_id[title] for title in rankings_to_eval[query]]
                              for query in query_ids]
  ordered_qrels = [[document_title_to_id[title] for title in qrels[query]]
                   for query in query_ids]
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
  encoded_docs = torch.stack([encode_glove_fs(glove, bm25.idf, doc_fs) for doc_fs in bm25.f])
  encoded_docs = encoded_docs / torch.norm(encoded_docs, dim=1).unsqueeze(1)
  for q, qml_ranking in progressbar(zip(tokenized_queries, ordered_rankings_to_eval),
                                    max_value=len(tokenized_queries)):
    doc_ids = qml_ranking if '--rerank' in sys.argv else None
    bm25_rankings.append(rank_bm25(bm25, q, average_idf=average_idf, doc_ids=doc_ids))
    glove_rankings.append(rank_glove(glove, bm25.idf, encoded_docs, q, doc_ids=doc_ids))
    rm3_rankings.append(rank_rm3(docs_lms, qml_ranking, q, doc_ids=doc_ids))
  print('indri:', metrics_at_k(ordered_rankings_to_eval, ordered_qrels, k))
  print('bm25:', metrics_at_k(bm25_rankings, ordered_qrels, k))
  print('glove:', metrics_at_k(glove_rankings, ordered_qrels, k))
  print('rm3:', metrics_at_k(rm3_rankings, ordered_qrels, k))

def main():
  if '--basic' in sys.argv:
    basic_eval()
  else:
    baselines_eval()

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
