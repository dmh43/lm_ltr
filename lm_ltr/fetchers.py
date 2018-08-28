from typing import List
import pymysql.cursors
import pickle

def get_documents_by_id(document_ids: List[int]) -> List[str]:
  pass

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
