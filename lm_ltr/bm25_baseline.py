from functools import partial
from multiprocessing import Pool, cpu_count
from six.moves import xrange
import pydash as _

from gensim.summarization.bm25 import BM25
from fastai.text import Tokenizer

import torch
import torch.nn as nn
import numpy as np

from lm_ltr.fetchers import read_cache, get_robust_test_queries, get_robust_rels, get_robust_documents
from lm_ltr.preprocessing import create_id_lookup

def effective_n_jobs(n_jobs):
  if n_jobs == 0:
    raise ValueError('n_jobs == 0 in Parallel has no meaning')
  elif n_jobs is None:
    return 1
  elif n_jobs < 0:
    n_jobs = max(cpu_count() + 1 + n_jobs, 1)
  return n_jobs

def _get_scores(bm25, document, average_idf):
  scores = []
  for index in xrange(bm25.corpus_size):
    score = bm25.get_score(document, index, average_idf)
    scores.append(score)
  return scores

query_lookup = read_cache('./robust_test_queries.pkl', get_robust_test_queries)
query_name_to_id = create_id_lookup(query_lookup.keys())
query_name_document_title_rels = read_cache('./robust_rels.pkl', get_robust_rels)
query_id_to_name = _.invert(query_name_to_id)
query_ids = range(len(query_id_to_name))
queries = [query_lookup[query_id_to_name[query_id]] for query_id in query_ids]
document_lookup = read_cache('./doc_lookup.pkl', get_robust_documents)
document_title_to_id = create_id_lookup(document_lookup.keys())
document_id_to_title = _.invert(document_title_to_id)
doc_ids = range(len(document_id_to_title))
documents = [document_lookup[document_id_to_title[doc_id]] for doc_id in doc_ids]
tokenizer = Tokenizer()
tokenized_documents = tokenizer.process_all(documents)
tokenized_queries = tokenizer.process_all(queries)
query_name_document_id_rels = _.map_values(query_name_document_title_rels,
                                           lambda doc_titles: [document_title_to_id[title]
                                                               for title in doc_titles
                                                               if title in document_title_to_id])
bm25 = BM25(tokenized_documents)
average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)
n_jobs = 20
n_processes = effective_n_jobs(n_jobs)

def get_score(document):
  return _get_scores(bm25, document, average_idf=average_idf)

if __name__ == "__main__":
  import ipdb
  import traceback
  import sys

  try:
    pool = Pool(n_processes)
    scores = pool.map(get_score, tokenized_queries)
    pool.close()
    pool.join()
    k = 10
    correct = 0
    num_relevant = 0
    num_rankings_considered = 0
    dcg = 0
    idcg = 0
    for query_id, doc_scores in enumerate(scores):
      query_name = query_id_to_name[query_id]
      rel_doc_ids = set(query_name_document_id_rels[query_name])
      topk_scores, topk_idxs = torch.topk(torch.tensor(doc_scores))
      sorted_scores, sort_idxs = torch.sort(topk_scores)
      ranked_doc_ids = topk_idxs[sort_idxs]
      for doc_rank, doc_id in enumerate(ranked_doc_ids):
        rel = doc_id in rel_doc_ids
        correct += rel
        dcg += (2 ** rel - 1) / np.log2(doc_rank + 2)
      num_relevant += len(rel_doc_ids)
      idcg += np.array([1.0/np.log2(rank + 2)
                        for rank in range(min(k, len(rel_doc_ids)))]).sum()
      if len(rel_doc_ids) > 0:
        num_rankings_considered += 1
    precision_k = correct / (k * num_rankings_considered)
    recall_k = correct / num_relevant
    ndcg = dcg / idcg
    print({'precision': precision_k, 'recall': recall_k, 'ndcg': ndcg})
  except: # pylint: disable=bare-except
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
  ipdb.post_mortem(tb)
