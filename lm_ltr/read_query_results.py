import pickle

from utils import append_at

def main():
  with open('./document_titles.pkl', 'rb') as fh:
    document_titles = pickle.load(fh)
  with open('./queries.pkl', 'rb') as fh:
    queries = pickle.load(fh)
  results = {}
  with open('./indri/query_result') as fh:
    line = fh.readline()
    if line:
      query_num, __, doc_num, doc_rank, doc_score, ___ = line.strip().split(' ')
      query = queries[query_num - 1]
      document_title = document_titles[doc_num - 1]
      append_at(results, query, {'rank': doc_rank, 'score': doc_score, 'document_title': document_title})
    else:
      with open('./indri_results.pkl', 'wb+') as fh:
        pickle.dump(results, fh)

if __name__ == "__main__": main()
