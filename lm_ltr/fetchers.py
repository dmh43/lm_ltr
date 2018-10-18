from functools import reduce
import re
from lxml import html

import pymysql.cursors
import pickle

import pydash as _

from .trec_doc_parse import parse_test_set, parse_qrels

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

def get_document_titles():
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
      el_cursor.execute("select title as document_title from pages")
      return list(el_cursor.fetchall())
  finally:
    el_connection.close()

def clean_text(text):
  return re.sub('\n', '', text.strip() + ' ')

def parse_xml_docs(path):
  with open(path, 'rb') as fh:
    text = str(fh.read().decode('latin-1'))
  text_to_parse = text if '<root>' in text else '<root>' + text + '</root>'
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
      el_cursor.execute("select title as document_title, content as document from pages")
      rows = el_cursor.fetchall()
      assert len(rows) == len(id_document_title_mapping)
      document_title_content_mapping = {row['document_title']: row['document'] for row in rows}
      documents = []
      for i in range(len(rows)):
        documents.append(document_title_content_mapping[id_document_title_mapping[i]])
      return documents
  finally:
    el_connection.close()

def get_supervised_raw_data(document_title_id_mapping, queries):
  rows = get_rows()
  queries_to_keep = set(queries)
  result = []
  ctr = 0
  for row in rows:
    if row['query'] not in queries_to_keep:
      ctr += 1
      continue
    if row['document_title'] in document_title_id_mapping:
      result.append({'query': row['query'],
                     'doc_id': document_title_id_mapping[row['document_title']]})
    else:
      ctr += 1
  print('skipped', ctr, 'supervised queries')
  return result

def get_weak_raw_data(id_query_mapping, queries):
  results = []
  with open('./indri/query_result') as fh:
    while True:
      line = fh.readline()
      if line:
        query_num, __, doc_num, doc_rank, doc_score, ___ = line.strip().split(' ')
        query_id = int(query_num) - 1
        query = id_query_mapping[query_id]
        if query_id not in queries: continue
        results.append({'query': query,
                        'doc_id': int(doc_num) - 1,
                        'score': float(doc_score),
                        'rank': int(doc_rank) - 1})
      else:
        return results

def read_query_result(query_name_to_id, document_title_to_id, queries, path='./indri/query_result'):
  results = []
  with open(path) as fh:
    while True:
      line = fh.readline()
      if line:
        query_name, __, doc_title, doc_rank, doc_score, ___ = line.strip().split(' ')
        query_id = query_name_to_id[query_name]
        if query_id not in queries: continue
        results.append({'query': queries[query_id],
                        'doc_id': document_title_to_id[doc_title],
                        'score': float(doc_score),
                        'rank': int(doc_rank) - 1})
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
  except (FileNotFoundError, pymysql.err.OperationalError):
    data = read_from_file(path)
  return data

def read_cache(path, fn):
  try:
    data = read_from_file(path)
  except FileNotFoundError:
    data = fn()
    write_to_file(path, data)
  return data

def get_robust_documents():
  doc_paths = ['./fbis', './la', './ft']
  return _.merge({}, *[parse_xml_docs(doc_path) for doc_path in doc_paths])

def get_robust_queries():
  return parse_test_set('./data/robust04/04.testset')

def get_robust_rels():
  return parse_qrels('./data/robust04/qrels.robust2004.txt')
