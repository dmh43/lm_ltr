import pydash as _

import os
import json

from lm_ltr.fetchers import get_robust_train_queries, read_cache

def main():
  path = './indri/robust_train_query_params_without_unks.xml'
  try:
    os.remove(path)
  except OSError:
    pass
  with open('./caches/pairwise_train_ranking_106756.json') as fh:
    query_ranking_pairs = json.load(fh)
    queries_by_tok_id, qml = zip(*query_ranking_pairs)
  parsed_queries, query_token_lookup = read_cache('./parsed_robust_queries_dict.json',
                                                  lambda: print('failed'))
  inv = _.invert(query_token_lookup)
  queries = [' '.join([inv[q] for q in query]) for query in queries_by_tok_id]
  with open(path, 'a+') as fh:
    fh.write('<parameters>\n')
    for query_name, query_text in enumerate(queries):
      query_name = str(query_name + 1)
      if len(query_text) == 0: continue
      fh.write('<query>\n')
      fh.write('<number>' + query_name + '</number>\n')
      fh.write('<text>\n')
      fh.write('#combine( ' + query_text + ' )\n')
      fh.write('</text>\n')
      fh.write('</query>\n')
    fh.write('</parameters>\n')


if __name__ == "__main__": main()
