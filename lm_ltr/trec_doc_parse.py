import pydash as _

from .utils import append_at

def parse_qrels(qrels_path):
  query_doc_id_rels = {}
  with open(qrels_path, 'r') as fh:
    for line in fh:
      query_num, __, doc_id, rel = line.strip().split(' ')
      if int(rel) == 1:
        append_at(query_doc_id_rels, query_num, doc_id)
    return query_doc_id_rels

def parse_test_set(test_set_path):
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
