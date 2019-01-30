import pydash as _

import sys
import gensim.summarization.bm25 as gensim_bm25
from gensim.models import TfidfModel, LsiModel
from fastai.text import Tokenizer, fix_html, spec_add_spaces, rm_useless_spaces
from progressbar import progressbar
import torch
import json

from lm_ltr.trec_doc_parse import parse_qrels
from lm_ltr.metrics import metrics_at_k
from lm_ltr.utils import append_at, name
from lm_ltr.fetchers import read_query_test_rankings, read_cache, get_robust_eval_queries, get_robust_rels, get_robust_documents_with_titles
from lm_ltr.preprocessing import create_id_lookup, handle_caps, clean_documents, tokens_to_indexes
from lm_ltr.embedding_loaders import get_glove_lookup
from lm_ltr.baselines import calc_docs_lms, rank_rm3, rank_glove, rank_bm25, rank_lsi, encode_glove_fs

def basic_eval():
  path = './indri/query_result_test'
  path = path + '_' + sys.argv[3] if len(sys.argv) >= 4 else path
  rankings_to_eval = read_query_test_rankings(path=path)
  qrels = parse_qrels()
  query_ids = list(qrels.keys())
  k = 10 if len(sys.argv) == 1 else int(sys.argv[1])
  document_title_to_id = read_cache('./document_title_to_id.json',
                                    lambda: print('failed'))
  ordered_rankings_to_eval = [[document_title_to_id[title] for title in rankings_to_eval[query]]
                              for query in query_ids]
  ordered_qrels = [[document_title_to_id[title] for title in qrels[query]]
                   for query in query_ids[:200]]
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
                                   lambda: tokenizer.process_all(clean_documents(documents)))
  tokenized_queries = tokenizer.process_all(clean_documents(queries))
  bm25 = gensim_bm25.BM25(tokenized_documents)
  # with open('./caches/106756_most_common_doc.json', 'r') as fh:
  #   doc_token_set = set(json.load(fh))
  # corpus, token_lookup = tokens_to_indexes(tokenized_documents,
  #                                          None,
  #                                          token_set=doc_token_set)
  # corpus = [[[token_lookup[term], f] for term, f in doc_fs.items()] for doc_fs in bm25.f]
  # tfidf = TfidfModel(corpus)
  # lsi = LsiModel(tfidf, id2word=_.invert(token_lookup), num_topics=300)
  glove_rankings = []
  # lsi_rankings = []
  glove = get_glove_lookup(embedding_dim=300, use_large_embed=True)
  encoded_docs = torch.stack([encode_glove_fs(glove, bm25.idf, doc_fs) for doc_fs in bm25.f])
  encoded_docs = encoded_docs / torch.norm(encoded_docs, dim=1).unsqueeze(1)
  for q, qml_ranking in progressbar(zip(tokenized_queries, ordered_rankings_to_eval),
                                    max_value=len(tokenized_queries)):
    doc_ids = qml_ranking[:k] if '--rerank' in sys.argv else None
    glove_rankings.append(rank_glove(glove, bm25.idf, encoded_docs, q, doc_ids=doc_ids))
    # lsi_rankings.append(rank_lsi(lsi, tfidf, [token_lookup[term] if term in token_lookup else 0 for term in q], doc_ids=doc_ids))
  print('indri:', metrics_at_k(ordered_rankings_to_eval, ordered_qrels, k))
  print('glove:', metrics_at_k(glove_rankings, ordered_qrels, k))
  # print('lsi:', metrics_at_k(lsi_rankings, ordered_qrels, k))

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
