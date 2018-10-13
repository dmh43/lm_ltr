import os
import pickle

import pydash as _

from .fetchers import get_raw_documents

def main():
  path = './indri/in'
  try:
    os.remove(path)
  except OSError:
    pass
  with open('./document_ids.pkl', 'rb') as fh:
    document_title_id_mapping = pickle.load(fh)
    id_document_title_mapping = {document_id: title for title, document_id in _.to_pairs(document_title_id_mapping)}
  documents = get_raw_documents(id_document_title_mapping)
  with open(path, 'a+') as fh:
    for i, document in enumerate(documents):
      fh.write('<DOC>\n')
      fh.write('<DOCNO>' + str(i + 1) + '</DOCNO>\n')
      fh.write('<TEXT>\n')
      fh.write(document + '\n')
      fh.write('</TEXT>\n')
      fh.write('</DOC>\n')


if __name__ == "__main__": main()
