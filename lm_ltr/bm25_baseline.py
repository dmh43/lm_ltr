from six.moves import xrange
import pydash as _
import pickle

from gensim.summarization.bm25 import BM25
from fastai.text import Tokenizer

import torch
import torch.nn as nn
import numpy as np

from lm_ltr.fetchers import read_cache, get_robust_test_queries, get_robust_rels, get_robust_documents
from lm_ltr.preprocessing import create_id_lookup
from lm_ltr.metrics import metrics_at_k

def _get_scores(bm25, document, average_idf):
  scores = []
  for index in xrange(bm25.corpus_size):
    score = bm25.get_score(document, index, average_idf)
    scores.append(score)
  return scores

def main():
  query_lookup = read_cache('./robust_test_queries.pkl', get_robust_test_queries)
  query_name_to_id = create_id_lookup(query_lookup.keys())
  query_name_document_title_rels = read_cache('./robust_rels.pkl', get_robust_rels)
  query_id_to_name = _.invert(query_name_to_id)
  query_ids = range(len(query_id_to_name))
  queries = [query_lookup[query_id_to_name[query_id]] for query_id in query_ids]
  document_lookup = read_cache('./doc_lookup.pkl', get_robust_documents)
  # num_doc_tokens_to_consider = 10000
  # document_lookup = _.map_values(document_lookup, lambda document: document[:num_doc_tokens_to_consider])
  document_title_to_id = create_id_lookup(document_lookup.keys())
  document_id_to_title = _.invert(document_title_to_id)
  doc_ids = range(len(document_id_to_title))
  documents = [document_lookup[document_id_to_title[doc_id]] for doc_id in doc_ids]
  tokenizer = Tokenizer()
  tokenized_documents = read_cache('tok_docs.pkl',
                                   lambda: tokenizer.process_all(documents))
  tokenized_queries = read_cache('tok_queries.pkl',
                                 lambda: tokenizer.process_all(queries))
  query_name_document_id_rels = _.map_values(query_name_document_title_rels,
                                             lambda doc_titles: [document_title_to_id[title]
                                                                 for title in doc_titles
                                                                 if title in document_title_to_id])
  try:
    with open('./bm25_scores_all_tokens.pkl', 'rb') as fh:
      scores = pickle.load(fh)
  except:
    bm25 = BM25(tokenized_documents)
    average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)
    scores = list(map(lambda document: _get_scores(bm25, document, average_idf=average_idf),
                      tokenized_queries))
    with open('./bm25_scores_all_tokens.pkl', 'wb+') as fh:
      pickle.dump(scores, fh)
  k = 10
  def get_rankings_to_judge():
    for doc_scores in scores:
      topk_scores, topk_idxs = torch.topk(torch.tensor(doc_scores), k)
      sorted_scores, sort_idxs = torch.sort(topk_scores, descending=True)
      ranked_doc_ids = topk_idxs[sort_idxs].tolist()
      yield ranked_doc_ids
  rel_doc_ids = [set(query_name_document_id_rels[query_id_to_name[query_id]])
                 for query_id in range(len(scores))
                 if (query_id in query_id_to_name) and (query_id_to_name[query_id] in query_name_document_id_rels)]
  print(metrics_at_k(get_rankings_to_judge(), rel_doc_ids, k))


if __name__ == "__main__": main()
