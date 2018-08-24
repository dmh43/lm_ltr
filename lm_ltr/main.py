from typing import List

import pydash as _
import pymysql.cursors
import torch
import torch.nn as nn

from raw_data_organizers import get_raw_train_test
from lm_scorer import LMScorer
from parsers import into_tokens
from train_model import train_model
from eval_model import eval_model

def count_unique_tokens(texts: List[str]) -> int:
  tokens: set = set()
  for text in texts:
    tokens = tokens.union(into_tokens(text))
  return len(tokens)

def get_model(num_query_terms: int, query_term_embed_len: int) -> nn.Module:
  query_term_embeds = nn.Embedding(num_query_terms, query_term_embed_len)
  return LMScorer(query_term_embeds)

def main():
  el_connection = pymysql.connect(host='localhost' ,
                                  user='danyhaddad',
                                  db='simplewiki' ,
                                  charset='utf8mb4',
                                  use_unicode=True,
                                  cursorclass=pymysql.cursors.DictCursor)

  try:
    with el_connection.cursor() as el_cursor:
      el_cursor.execute("SET NAMES utf8mb4;")
      el_cursor.execute("SET CHARACTER SET utf8mb4;")
      el_cursor.execute("SET character_set_connection=utf8mb4;")
      el_cursor.execute("select mention as query, entity as title, pages.content as document from entity_mentions_text  inner join pages on pages.id=entity_mentions_text.page_id where mention not like concat('%', entity, '%') and entity not like concat('%', mention,'%')")
      raw_data = get_raw_train_test(el_cursor.fetchall())
      query_term_embed_len = 100
      num_query_terms = count_unique_tokens(raw_data['train_queries'])
      model = get_model(num_query_terms, query_term_embed_len)
      train_model(model, raw_data)
      eval_model(model, raw_data)
  finally:
    el_connection.close()


if __name__ == "__main__": main()
