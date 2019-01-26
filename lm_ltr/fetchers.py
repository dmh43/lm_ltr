import glob
from functools import reduce
import re
from lxml import html
import os

import pymysql.cursors
import pickle
import json
import shelve
from dbm import error as DbmFileError
from collections import defaultdict

import pydash as _

from .trec_doc_parse import parse_test_set, parse_qrels
from .utils import append_at
from .shelve_array import ShelveArray
from .snorkel_helper import get_pairwise_bins
from .types import TargetInfo, QueryPairwiseBinsByRanker

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
    if hasattr(texts, 'text_content'):
      text = clean_text(texts.text_content())
    else:
      text = reduce(lambda acc, p: acc + clean_text(p.text_content()),
                    texts.getchildren(),
                    '')
    docs[(doc.find('docno')).text.strip()] = text
  return docs

def parse_xml_docs_for_titles(path):
  with open(path, 'rb') as fh:
    text = str(fh.read().decode('latin-1'))
  text_to_parse = text if '<root>' in text else '<root>' + text + '</root>'
  tree = html.fromstring(text_to_parse)
  docs = {}
  for doc in tree:
    texts = doc.find('headline') or doc.find('header')
    if texts is None: continue
    if hasattr(texts, 'text_content'):
      text = clean_text(texts.text_content())
    else:
      text = reduce(lambda acc, p: acc + clean_text(p.text_content()),
                    texts.getchildren(),
                    '')
    docs[(doc.find('docno')).text.strip()] = text
  return docs


def parse_xml_docs_and_titles(path):
  with open(path, 'rb') as fh:
    text = str(fh.read().decode('latin-1'))
  text_to_parse = text if '<root>' in text else '<root>' + text + '</root>'
  tree = html.fromstring(text_to_parse)
  docs = {}
  for doc in tree:
    def _get_text(node):
      results = []
      if len(node.getchildren()) == 0 and node.tag not in ['docno', 'docid', 'date', 'date1', 'ht']:
        results.append(clean_text(node.text_content()))
      else:
        for child in node.getchildren():
          results.extend(_get_text(child))
      return results
    docs[(doc.find('docno')).text.strip()] = '\n'.join(_get_text(doc))
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

def get_weak_raw_data(id_query_mapping, queries, path='./indri/query_result'):
  results = []
  with open(path) as fh:
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

def read_query_test_rankings(path='./indri/query_result_test'):
  rankings = {}
  with open(path) as fh:
    for line in fh:
      query_name, __, doc_title, doc_rank, doc_score, ___ = line.strip().split(' ')
      append_at(rankings, query_name, doc_title)
    return rankings

def read_query_result(query_name_to_id, document_title_to_id, queries, path='./indri/query_result'):
  results = []
  with open(path) as fh:
    while True:
      line = fh.readline()
      if line:
        query_name, __, doc_title, doc_rank, doc_score, ___ = line.strip().split(' ')
        if query_name not in query_name_to_id: continue
        if doc_title not in document_title_to_id: continue
        query_id = query_name_to_id[query_name]
        score = float(doc_score)
        if query_id not in queries:
          query_id = str(query_id)
          if query_id not in queries: continue
        results.append({'query': queries[query_id],
                        'doc_id': document_title_to_id[doc_title],
                        'score': score,
                        'rank': int(doc_rank) - 1})
      else:
        return results

def get_query_str_to_pairwise_bins(query_name_to_id, document_title_to_id, queries, path, limit=None):
  rankings_by_query = defaultdict(list)
  with open(path) as fh:
    while True:
      if limit is not None and len(rankings_by_query) >= limit: break
      line = fh.readline()
      if line:
        query_name, __, doc_title, __, __, ___ = line.strip().split(' ')
        if query_name not in query_name_to_id: continue
        if doc_title not in document_title_to_id: continue
        query_id = query_name_to_id[query_name]
        if query_id not in queries:
          query_id = str(query_id)
          if query_id not in queries: continue
        rankings_by_query[str(queries[query_id])[1:-1]].append(document_title_to_id[doc_title])
      else:
        return _.map_values(dict(rankings_by_query), get_pairwise_bins)

def get_ranker_query_str_to_pairwise_bins(query_name_to_id,
                                          document_title_to_id,
                                          queries,
                                          limit=None,
                                          rankers=('qml', 'bm25', 'tfidf', 'rm3'),
                                          path='./indri/query_result') -> QueryPairwiseBinsByRanker:
  ranker_name_to_suffix = {'qml': '', 'bm25': '_okapi', 'tfidf': '_tfidf', 'rm3': '_fb'}
  return {ranker: get_query_str_to_pairwise_bins(query_name_to_id,
                                                 document_title_to_id,
                                                 queries,
                                                 path + ranker_name_to_suffix[ranker],
                                                 limit=limit)
          for ranker in rankers}

def get_query_str_to_rankings(query_name_to_id, document_title_to_id, queries, path, limit=None):
  rankings_by_query = defaultdict(list)
  with open(path) as fh:
    while True:
      if limit is not None and len(rankings_by_query) >= limit: break
      line = fh.readline()
      if line:
        query_name, __, doc_title, __, __, ___ = line.strip().split(' ')
        if query_name not in query_name_to_id: continue
        if doc_title not in document_title_to_id: continue
        query_id = query_name_to_id[query_name]
        if query_id not in queries:
          query_id = str(query_id)
          if query_id not in queries: continue
        rankings_by_query[str(queries[query_id])[1:-1]].append(document_title_to_id[doc_title])
      else:
        return dict(rankings_by_query)

def get_ranker_query_str_to_rankings(query_name_to_id,
                                     document_title_to_id,
                                     queries,
                                     limit=None,
                                     rankers=('qml', 'bm25', 'tfidf', 'rm3'),
                                     path='./indri/query_result') -> QueryPairwiseBinsByRanker:
  ranker_name_to_suffix = {'qml': '', 'bm25': '_okapi', 'tfidf': '_tfidf', 'rm3': '_fb'}
  return {ranker: get_query_str_to_rankings(query_name_to_id,
                                            document_title_to_id,
                                            queries,
                                            path + ranker_name_to_suffix[ranker],
                                            limit=limit)
          for ranker in rankers}

def write_to_file(path, rows):
  if '.pkl' in path:
    with open(path, 'wb+') as fh:
      pickle.dump(rows, fh)
  else:
    with open(path, 'w+') as fh:
      json.dump(rows, fh)

def read_from_file(path):
  if '.pkl' in path:
    with open(path, 'rb') as fh:
      return pickle.load(fh)
  else:
    with open(path, 'r') as fh:
      return json.load(fh)

def read_or_cache(path, fn):
  try:
    data = fn()
    write_to_file(path, data)
  except (FileNotFoundError, pymysql.err.OperationalError):
    data = read_from_file(path)
  return data

def read_cache(path, fn):
  path = './caches/' + path
  try:
    data = read_from_file(path)
  except FileNotFoundError:
    data = fn()
    write_to_file(path, data)
  return data

def get_robust_documents():
  doc_paths = ['./fbis', './la', './ft']
  return _.merge({}, *[parse_xml_docs(doc_path) for doc_path in doc_paths])

def get_robust_documents_with_titles():
  doc_paths = ['./fbis', './la', './ft']
  return _.merge({}, *[parse_xml_docs_and_titles(doc_path) for doc_path in doc_paths])

def get_robust_eval_queries():
  return parse_test_set('./data/robust04/04.testset')

def get_robust_train_queries():
  def _clean_string(query):
    cleaned = re.sub(r'[^a-zA-Z0-9 ]', '', query)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return re.sub(r'-', ' ', cleaned).strip()
  def _is_invalid(string):
    for invalid_str in ['http', 'www.', '.com', '.net', '.org', '.edu']:
      if invalid_str in string: return True
    return False
  paths = glob.glob('./data/aol/queries/user-ct-test-collection-[0-9][0-9].txt')
  queries = {}
  seen_queries = set()
  for path in paths:
    with open(path) as fh:
      for i, line in enumerate(fh):
        if i == 0: continue
        query_str = line.strip().split('\t')[1]
        if _is_invalid(query_str): continue
        query_str = _clean_string(query_str)
        if len(query_str) == 0: continue
        if query_str in seen_queries: continue
        query_name = str(len(queries) + 1)
        queries[query_name] = query_str
        seen_queries.add(query_str)
  return queries

def get_robust_rels():
  return parse_qrels('./data/robust04/qrels.robust2004.txt')

def get_wiki_pages():
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
      el_cursor.execute("select title, content from pages")
      return {row['title']: row['content'] for row in el_cursor.fetchall()}
  finally:
    el_connection.close()

def get_mention_page_title_pairs():
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
      el_cursor.execute("select distinct mention_id, entity from entity_mentions_text")
      return [[row['mention_id'], row['entity']] for row in el_cursor.fetchall()]
  finally:
    el_connection.close()

def get_wiki_mentions():
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
      el_cursor.execute("select distinct mention_id, mention from entity_mentions_text")
      return {row['mention_id']: row['mention'] for row in el_cursor.fetchall()}
  finally:
    el_connection.close()
