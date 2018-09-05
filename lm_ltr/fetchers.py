import pymysql.cursors
import pickle

import pydash as _

def get_rows():
  el_connection = pymysql.connect(host='localhost' ,
                                  user='danyhaddad',
                                  db='el' ,
                                  charset='utf8mb4',
                                  use_unicode=True,
                                  cursorclass=pymysql.cursors.DictCursor)
  try:
    with el_connection.cursor() as el_cursor:
      el_cursor.execute("SET NAMES utf8mb4;")
      el_cursor.execute("SET CHARACTER SET utf8mb4;")
      el_cursor.execute("SET character_set_connection=utf8mb4;")
      el_cursor.execute("select distinct mention as query, entity as document_title from entity_mentions_text where mention not like concat('%', entity, '%') and entity not like concat('%', mention,'%')")
      return list(el_cursor.fetchall())
  finally:
    el_connection.close()

def get_raw_documents(id_document_title_mapping):
  el_connection = pymysql.connect(host='localhost' ,
                                  user='danyhaddad',
                                  db='el' ,
                                  charset='utf8mb4',
                                  use_unicode=True,
                                  cursorclass=pymysql.cursors.DictCursor)
  try:
    with el_connection.cursor() as el_cursor:
      el_cursor.execute("SET NAMES utf8mb4;")
      el_cursor.execute("SET CHARACTER SET utf8mb4;")
      el_cursor.execute("SET character_set_connection=utf8mb4;")
      el_cursor.execute("select document_title, content as document from pages")
      rows = el_cursor.fetchall()
      document_title_content_mapping = {row['document_title']: row['content'] for row in rows}
      assert len(id_document_title_mapping) == len(rows), "ID document_title mapping should be 1-1"
      documents = []
      for i in range(len(rows)):
        documents.append(document_title_content_mapping[id_document_title_mapping[i]])
      return documents
  finally:
    el_connection.close()

def get_supervised_raw_data(document_title_id_mapping, queries):
  rows = get_rows()
  queries_to_keep = set(queries)
  return [{'query': row['query'],
           'document_id': document_title_id_mapping[row['document_title']]} for row in rows if row['query'] in queries_to_keep]

def get_weak_raw_data(id_query_mapping, queries):
  results = []
  with open('./indri/query_result') as fh:
    while True:
      line = fh.readline()
      if line:
        query_num, __, doc_num, doc_rank, doc_score, ___ = line.strip().split(' ')
        query = id_query_mapping[query_num - 1]
        if query not in queries: continue
        results.append({'query': query,
                        'document_id': doc_num - 1})
      else:
        return results

def write_to_file(path, rows):
  with open(path, 'wb') as fh:
    pickle.dump(rows, fh)

def read_from_file(path):
  with open(path, 'rb') as fh:
    return pickle.load(fh)

def read_or_cache(path, fn):
  try:
    data = fn()
    write_to_file(path, data)
  except:
    data = read_from_file(path)
  return data
