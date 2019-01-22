import pydash as _

from .utils import append_at

def parse_qrels(qrels_path='./data/robust04/qrels.robust2004.txt'):
  query_doc_id_rels = {}
  with open(qrels_path, 'r') as fh:
    for line in fh:
      query_num, __, doc_id, rel = line.strip().split()
      if int(rel) == 1:
        if any([name in doc_id for name in ['FBIS', 'FT', 'LA']]):
          append_at(query_doc_id_rels, query_num, doc_id)
    return query_doc_id_rels

def parse_test_set(test_set_path):
  with open(test_set_path, 'r') as fh:
    queries = {}
    current_query = None
    check_next_line = False
    for line in fh:
      line = line.strip()
      if '<num>' in line:
        current_query = line.split(' ')[-1]
      elif '<title>' in line:
        query = ' '.join(line.split(' ')[1:]).strip()
        if query != '':
          queries[current_query] = query
          current_query = None
        else:
          check_next_line = True
      elif check_next_line:
        check_next_line = False
        query = line.strip()
        queries[current_query] = query
        current_query = None
    return queries
