import pickle

from .fetchers import get_rows, get_document_titles


def main():
  document_title_id_mapping = {row['document_title']: i for i, row in enumerate(get_document_titles())}
  query_id_mapping = {}
  ctr = 0
  for row in get_rows():
    if row['document_title'] in document_title_id_mapping:
      query_id_mapping[row['query']] = len(query_id_mapping)
    else:
      ctr += 1
  with open('./document_ids.pkl', 'wb+') as fh:
    pickle.dump(document_title_id_mapping, fh)
  with open('./query_ids.pkl', 'wb+') as fh:
    pickle.dump(query_id_mapping, fh)
  print(ctr, 'queries ignored')

if __name__ == "__main__": main()
