import os
import pickle
import re

from fetchers import get_rows

def clean_string(query):
  cleaned = re.sub('[^a-zA-Z0-9]', '', query)
  return re.sub('-', ' ', cleaned).strip()

def main():
  path = './indri/query_params.xml'
  try:
    os.remove(path)
  except OSError:
    pass
  with open('./query_ids.pkl', 'rb') as fh:
    query_id_mapping = pickle.load(fh)
  rows = get_rows()
  with open(path, 'a+') as fh:
    fh.write('<parameters>\n')
    for row in rows:
      if row['query'] not in query_id_mapping: continue
      query_id = query_id_mapping[row['query']]
      query = clean_string(row['query'])
      if len(query) == 0: continue
      fh.write('<query>\n')
      fh.write('<number>' + str(query_id + 1) + '</number>\n')
      fh.write('<text>\n')
      fh.write('#combine( ' + query + ' )\n')
      fh.write('</text>\n')
      fh.write('</query>\n')
    fh.write('</parameters>\n')


if __name__ == "__main__": main()
