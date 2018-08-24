from typing import List

def _get_candidate_document_ids_bm25(query: str) -> List[int]:
  pass

def get_candidate_document_ids(query: str, method: str='BM25') -> List[int]:
  if method == 'BM25':
    return _get_candidate_document_ids_bm25(query)
  else:
    raise NotImplementedError('Only BM25 supported')
