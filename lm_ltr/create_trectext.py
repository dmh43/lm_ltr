import os
import pickle

from fetchers import get_rows

def main():
  path = './indri/in'
  try:
    os.remove(path)
  except OSError:
    pass
  rows = get_rows()
  with open('./document_ids.pkl', 'rb') as fh:
    document_title_id_mapping = pickle.load(fh)
  with open(path, 'a+') as fh:
    for row in rows:
      fh.write('<DOC>\n')
      fh.write('<DOCNO>' + str(document_title_id_mapping[row['document_title']] + 1) + '</DOCNO>\n')
      fh.write('<TEXT>\n')
      fh.write(row['document'] + '\n')
      fh.write('</TEXT>\n')
      fh.write('</DOC>\n')


if __name__ == "__main__": main()
