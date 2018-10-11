from functools import reduce
import re
import xml.etree.ElementTree as ET

from utils import append_at

def parse_xml_docs(path):
  with open(path, 'r') as fh:
    text = fh.read()
  tree = ET.fromstring('<root>' + text + '</root>')
  docs = {}
  for doc in tree:
    text = reduce(lambda acc, p: acc + ' ' + re.sub('\n', '', p.text.strip()),
                  doc.find('TEXT'),
                  '')
    docs[doc.find('DOCID').strip()] = text
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

def parse_query_doc_rels(qrels_path, test_set_path):
  query_doc_id_rels = _parse_qrels(qrels_path)
  queries = _parse_test_set(test_set_path)
  query_doc_rels = {}
  # for query in queries:
  #   tokenize and create ranking pairs
