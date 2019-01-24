import os
import sys

from lm_ltr.fetchers import get_robust_train_queries

def main():
  if '--no-combine' in sys.argv:
    path = './indri/robust_train_query_params_no_combine.xml'
  else:
    path = './indri/robust_train_query_params.xml'
  try:
    os.remove(path)
  except OSError:
    pass
  query_name_to_text = get_robust_train_queries()
  with open(path, 'a+') as fh:
    fh.write('<parameters>\n')
    for query_name, query_text in query_name_to_text.items():
      if len(query_text) == 0: continue
      fh.write('<query>\n')
      fh.write('<number>' + query_name + '</number>\n')
      fh.write('<text>\n')
      if '--no-combine' in sys.argv:
        fh.write(query_text + '\n')
      else:
        fh.write('#combine( ' + query_text + ' )\n')
      fh.write('</text>\n')
      fh.write('</query>\n')
    fh.write('</parameters>\n')


if __name__ == "__main__": main()
