import pickle

from fetchers import get_rows


def main():
  rows = get_rows()
  document_title_id_mapping = {row['document_title']: i for i, row in enumerate(rows)}
  query_id_mapping = {row['query']: i for i, row in enumerate(rows)}
  with open('./document_ids.pkl', 'wb+') as fh:
    pickle.dump(document_title_id_mapping, fh)
  with open('./query_ids.pkl', 'wb+') as fh:
    pickle.dump(query_id_mapping, fh)

if __name__ == "__main__": main()
