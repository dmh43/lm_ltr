from functools import reduce
import re
from lxml import html

import pydash as _

from .utils import append_at
from .preprocessing import preprocess_texts
from .fetchers import read_cache

def clean_text(text):
  return re.sub('\n', '', text.strip() + ' ')

def _parse_xml_docs(path):
  with open(path, 'rb') as fh:
    text = str(fh.read().decode('latin-1'))
  text_to_parse = text if '<root>' in text else '<root>' + text + '</root>'
  if len(text_to_parse) > 100000:
    tree = read_cache('./tree_parse_' + re.sub('/', '', path) + '.pkl',
                      lambda: html.fromstring(text_to_parse))
  else:
    tree = html.fromstring(text_to_parse)
  docs = {}
  for doc in tree:
    texts = doc.find('text')
    if texts is None: continue
    if len(texts.getchildren()) == 0:
      text = clean_text(texts.text_content())
    else:
      text = reduce(lambda acc, p: acc + clean_text(p.text_content()),
                    texts.getchildren(),
                    '')
    docs[(doc.find('docno')).text.strip()] = text
  return docs

def _parse_qrels(qrels_path):
  query_doc_id_rels = {}
  with open(qrels_path, 'r') as fh:
    for line in fh:
      query_num, __, doc_id, rel = line.strip().split(' ')
      if int(rel) == 1:
        append_at(query_doc_id_rels, query_num, doc_id)
    return query_doc_id_rels

def _parse_test_set(test_set_path):
  with open(test_set_path, 'r') as fh:
    queries = {}
    current_query = None
    for line in fh:
      line = line.strip()
      if '<num>' in line:
        current_query = line.split(' ')[-1]
      elif '<title>' in line:
        queries[current_query] = ' '.join(line.split(' ')[1:])
        current_query = None
    return queries

def map_docs(docs, doc_id_lookup, document_token_lookup=None):
  doc_ids = sorted(list(doc_id_lookup.values()))
  old_doc_id_lookup = _.invert(doc_id_lookup)
  contents = [docs[old_doc_id_lookup[doc_id]] for doc_id in doc_ids]
  indexed_docs, document_token_lookup = preprocess_texts(contents, token_lookup=document_token_lookup)
  return indexed_docs

def map_queries(queries, query_token_lookup=None):
  query_ids = list(queries.keys())
  query_strings = [queries[query_id] for query_id in query_ids]
  indexed_queries, query_token_lookup = preprocess_texts(query_strings, token_lookup=query_token_lookup)
  return dict(zip(query_ids, indexed_queries))

def load_robust04_documents_lookup(doc_paths):
  return _.merge({}, *[_parse_xml_docs(doc_path) for doc_path in doc_paths])

def get_robust_queries_lookup():
  query_doc_no_rels = _parse_qrels(qrels_path)
  queries = _parse_test_set(test_set_path)

def parse_robust(query_token_lookup,
                 document_token_lookup,
                 qrels_path,
                 test_set_path,
                 doc_paths,
                 doc_first_index=0):
  docs = _.merge({}, *[_parse_xml_docs(doc_path) for doc_path in doc_paths])
  doc_id_lookup = dict(zip(docs.keys(), range(doc_first_index, doc_first_index + len(docs))))
  indexed_docs = map_docs(docs, doc_id_lookup, document_token_lookup)
  query_doc_no_rels = _parse_qrels(qrels_path)
  queries = _parse_test_set(test_set_path)
  index_queries_by_id = map_queries(queries, query_token_lookup)
  robust_data = []
  for query in queries:
    if query not in query_doc_no_rels: continue
    query_indexes = index_queries_by_id[query]
    for doc_no in query_doc_no_rels[query]:
      if doc_no not in doc_id_lookup: continue
      robust_data.append({'query': query_indexes, 'doc_id': doc_id_lookup[doc_no]})
  return robust_data, indexed_docs
