from random import shuffle

import pydash as _

def get_raw_train_test(rows, size=0.8):
  num_train_rows = int(size * len(rows))
  documents = []
  query_label_lookup = {}
  for row in rows:
    if query_label_lookup.get(row['query']) is not None:
      query_label_lookup[row['query']] = len(documents)
      documents.append(row['document'])
  samples = _.to_pairs(query_label_lookup)
  shuffle(samples)
  queries, labels = zip(*samples)
  return {'train_queries': queries[:num_train_rows],
          'test_queries': queries[num_train_rows:],
          'train_labels': labels[:num_train_rows],
          'test_labels': labels[num_train_rows:],
          'documents': documents}
