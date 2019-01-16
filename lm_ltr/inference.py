import torch

def rank_documents(query, documents, ranker, doc_ids=None, k=None):
  num_documents = len(documents)
  if num_documents < k: raise ValueError('Not enough documents to rank')
  doc_ids = doc_ids if doc_ids is not None else torch.arange(num_documents)
  k = k if k is not None else num_documents
  ranking_ids_for_batch = torch.squeeze(ranker(torch.unsqueeze(query, 0),
                                               documents,
                                               smooth=0.0,
                                               k=k))
  return doc_ids[ranking_ids_for_batch]
