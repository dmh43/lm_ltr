import sys
import os
import re

from lm_ltr.fetchers import get_robust_eval_queries

def clean_string(query):
  cleaned = re.sub('[^a-zA-Z0-9 ]', '', query)
  return re.sub('-', ' ', cleaned).strip()

def main():
  path = './indri/robust_test_query_params.xml'
  try:
    os.remove(path)
  except OSError:
    pass
  query_name_to_text = get_robust_eval_queries()
  with open(path, 'a+') as fh:
    fh.write('<parameters>\n')
    for query_name, query_text in query_name_to_text.items():
      cleaned_query = clean_string(query_text)
      if len(query_text) == 0: continue
      fh.write('<query>\n')
      fh.write('<number>' + query_name + '</number>\n')
      fh.write('<text>\n')
      if '--no-combine' in sys.argv:
        fh.write(cleaned_query + '\n')
      else:
        fh.write('#combine( ' + cleaned_query + ' )\n')
      fh.write('</text>\n')
      fh.write('</query>\n')
    fh.write('</parameters>\n')


if __name__ == "__main__": main()
