from six.moves import xrange
import pydash as _
import json

from gensim.summarization.bm25 import BM25
from fastai.text import Tokenizer

import torch
import torch.nn as nn
import numpy as np

from lm_ltr.fetchers import read_cache, get_robust_test_queries, get_robust_rels, get_robust_documents
from lm_ltr.preprocessing import create_id_lookup
from lm_ltr.metrics import metrics_at_k

def main():
  document_lookup = read_cache('./doc_lookup.json', get_robust_documents)
  document_title_to_id = create_id_lookup(document_lookup.keys())
  document_id_to_title = _.invert(document_title_to_id)
  doc_ids = range(len(document_id_to_title))
  documents = [document_lookup[document_id_to_title[doc_id]] for doc_id in doc_ids]
  tokenizer = Tokenizer()
  tokenized_documents = read_cache('tok_docs.json',
                                   lambda: tokenizer.process_all(documents))
  bm25 = BM25(tokenized_documents)
  with open('./doc_word_idf.json', 'w+') as fh:
    json.dump(bm25.idf, fh)


if __name__ == "__main__": main()
