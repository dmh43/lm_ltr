from typing import List
import math
import random
import pickle

import pydash as _
import pymysql.cursors
from fastai.text import Tokenizer
import torch
import torch.nn as nn

from raw_data_organizers import get_raw_train_test
from lm_scorer import LMScorer
from train_model import train_model
from eval_model import eval_model
from embedding_loaders import get_glove_lookup
from preprocessing import preprocess_texts

def count_unique_tokens(texts: List[str]) -> int:
  tokenizer = Tokenizer()
  tokens: set = set()
  for text in texts:
    tokens = tokens.union(tokenizer.proc_text(text))
  return len(tokens)

def get_embedding(glove_lookup, token_index_lookup, num_tokens, embed_len):
  token_embed_weights = nn.Parameter(torch.Tensor(num_tokens,
                                                  embed_len))
  token_embed_weights.data.normal_(0, 1.0/math.sqrt(embed_len))
  for token, index in token_index_lookup.items():
    if token in glove_lookup:
      token_embed_weights.data[index] = glove_lookup[token]
  embedding = nn.Embedding(num_tokens, embed_len)
  embedding.weight = token_embed_weights
  return embedding

def get_model(query_token_embed_len: int,
              document_token_embed_len: int,
              query_token_index_lookup: dict,
              document_token_index_lookup: dict) -> nn.Module:
  glove_lookup = get_glove_lookup()
  num_query_tokens = len(query_token_index_lookup)
  num_document_tokens = len(document_token_index_lookup)
  query_token_embeds = get_embedding(glove_lookup,
                                     query_token_index_lookup,
                                     num_query_tokens,
                                     query_token_embed_len)
  document_token_embeds = get_embedding(glove_lookup,
                                        document_token_index_lookup,
                                        num_document_tokens,
                                        document_token_embed_len)
  return LMScorer(query_token_embeds, document_token_embeds)

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
      el_cursor.execute("select mention as query, entity as title, pages.content as document from entity_mentions_text  inner join pages on pages.id=entity_mentions_text.page_id where mention not like concat('%', entity, '%') and entity not like concat('%', mention,'%')")
      return el_cursor.fetchall()
  finally:
    el_connection.close()

def write_rows_to_file(path, rows):
  with open(path, 'wb') as fh:
    pickle.dump(rows, fh)

def read_rows_from_file(path):
  with open(path, 'rb') as fh:
    return pickle.load(fh)

def get_negative_samples(num_query_tokens, num_negative_samples, max_len=4):
  result = []
  for i in range(num_negative_samples):
    query_len = random.randint(1, max_len)
    query = random.choices(range(2, num_query_tokens), k=query_len)
    result.append(query)
  return result

def preprocess_raw_data(raw_data):
  raw_documents, raw_train_queries, train_document_ids, raw_test_queries, test_document_ids = _.map_(['documents',
                                                                                                      'train_queries',
                                                                                                      'train_document_ids',
                                                                                                      'test_queries',
                                                                                                      'test_document_ids'],
                                                                                                     lambda key: raw_data[key])
  num_negative_train_samples = len(raw_train_queries) * 10
  num_negative_test_samples = len(raw_test_queries) * 10
  documents, document_token_lookup = preprocess_texts(raw_documents)
  processed_train_queries, query_token_lookup = preprocess_texts(raw_train_queries)
  processed_test_queries, __ = preprocess_texts(raw_test_queries, query_token_lookup)
  train_queries = processed_train_queries + get_negative_samples(len(query_token_lookup),
                                                                 num_negative_train_samples)
  test_queries = processed_test_queries + get_negative_samples(len(query_token_lookup),
                                                               num_negative_test_samples)
  train_labels = [1] * len(processed_train_queries) + [0] * num_negative_train_samples
  test_labels = [1] * len(processed_test_queries) + [0] * num_negative_test_samples
  return documents, document_token_lookup, query_token_lookup, train_queries, train_labels, test_queries, test_labels

def main():
  print('Getting dataset')
  try:
    rows = get_rows()
    write_rows_to_file('./rows', rows)
  except:
    rows = read_rows_from_file('./rows')
  raw_data = get_raw_train_test(rows)
  print('Preprocessing datasets')
  documents, document_token_lookup, query_token_lookup, train_queries, train_labels, test_queries, test_labels = preprocess_raw_data(raw_data)
  query_token_embed_len = 100
  document_token_embed_len = 100
  model = get_model(query_token_embed_len,
                    document_token_embed_len,
                    query_token_lookup,
                    document_token_lookup)
  train_model(model, documents, train_queries, train_labels, test_queries, test_labels)
  eval_model(model, raw_data)


if __name__ == "__main__":
  import ipdb
  import traceback
  import sys

  try:
    main()
  except: # pylint: disable=bare-except
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
ipdb.post_mortem(tb)
