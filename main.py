import pydash as _
import pymysql.cursors

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
      el_cursor.execute("select mention as query, entity as title, pages.content as text from entity_mentions_text  inner join pages on pages.id=entity_mentions_text.page_id where mention not like concat('%', entity, '%') and entity not like concat('%', mention,'%')")
      text_lookup = {}
      title_lookup = {}
      test = []
      for row_num, row in enumerate(el_cursor.fetchall()):
        title_lookup[row_num] = row['title']
        text_lookup[row['title']] = row['text']
        test.append({'query': row['query'],
                     'document_id': row_num,
                     'text': row['text']})
      eval_model(model, test)
  finally:
    el_connection.close()


if __name__ == "__main__": main()
