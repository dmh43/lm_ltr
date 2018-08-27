from random import shuffle

import pydash as _

def get_raw_train_test(rows, size=0.8):
  documents = []
  query_document_id_lookup = {}
  for row in rows:
    if query_document_id_lookup.get(row['query']) is None:
      query_document_id_lookup[row['query']] = len(documents)
      documents.append(row['document'])
  samples = _.to_pairs(query_document_id_lookup)
  num_train_samples = int(size * len(samples))
  shuffle(samples)
  queries, document_ids = list(zip(*samples))
  return {'train_queries': queries[:num_train_samples],
          'test_queries': queries[num_train_samples:],
          'train_document_ids': document_ids[:num_train_samples],
          'test_document_ids': document_ids[num_train_samples:],
          'documents': documents}
