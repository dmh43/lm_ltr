from functools import reduce
import re
import xml.etree.ElementTree as ET

import pydash as _

from utils import append_at
from preprocessing import preprocess_texts

def _parse_xml_docs(path):
  with open(path, 'rb') as fh:
    text = str(fh.read().decode('latin-1'))
  parser = ET.XMLParser(encoding='latin-1')
  tree = ET.fromstring('<root>' + text + '</root>', parser=parser)
  docs = {}
  for doc in tree:
    texts = doc.find('TEXT')
    if texts is None: continue
    text = reduce(lambda acc, p: acc + re.sub('\n', '', p.text.strip() + ' '),
                  list(texts),
                  '')
    docs[doc.find('DOCNO').text.strip()] = text
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

def map_docs(docs, doc_id_lookup):
  doc_ids = list(doc_id_lookup.values())
  old_doc_id_lookup = _.invert(doc_id_lookup)
  contents = [docs[old_doc_id_lookup[doc_id]] for doc_id in doc_ids]
  indexed_docs, doc_token_lookup = preprocess_texts(contents)
  return dict(zip(doc_ids, indexed_docs))

def parse_robust(query_token_lookup, qrels_path, test_set_path, doc_paths):
  docs = _.merge({}, *[_parse_xml_docs(doc_path) for doc_path in doc_paths])
  doc_id_lookup = dict(zip(docs.keys(), range(len(docs))))
  docs_by_index = map_docs(docs, doc_id_lookup)
  query_doc_no_rels = _parse_qrels(qrels_path)
  queries = _parse_test_set(test_set_path)
  queries_by_index = _.map_values(queries,
                                  lambda query: query_token_lookup.get(query) or query_token_lookup['<unk>'])
  query_doc_rels = []
  for query in queries:
    if query not in query_doc_no_rels: continue
    query_indexes = queries_by_index[query]
    rel_doc_ids = [doc_id_lookup[doc_no] for doc_no in query_doc_no_rels[query] if doc_no in doc_id_lookup]
    query_doc_rels.append([query_indexes, rel_doc_ids])
  return query_doc_rels, docs_by_index
