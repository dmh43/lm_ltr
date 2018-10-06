from functools import reduce
import pydash as _

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler

import scipy.sparse as sp
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer

from preprocessing import collate_query_samples, collate_query_pairwise_samples, to_query_rankings_pairs, pad_to_max_len, all_ones, score, inv_log_rank, inv_rank, exp_score

class QueryDataset(Dataset):
  def __init__(self, documents, data, rel_method=score, num_doc_tokens=100):
    self.documents = documents
    self.data = data
    self.rel_method = rel_method
    self.rankings = to_query_rankings_pairs(data)
    self.num_doc_tokens = num_doc_tokens

  def _get_document(self, doc_idx):
    return self.documents[self.data[doc_idx]['document_id']][:self.num_doc_tokens]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return ((self.data[idx]['query'], self._get_document(idx)),
            self.rel_method(self.data[idx]))

class RankingDataset(Dataset):
  def __init__(self,
               documents,
               rankings,
               query_document_token_mapping,
               num_doc_tokens=100,
               k=10):
    self.rankings = rankings
    self.documents = torch.tensor(pad_to_max_len(documents), dtype=torch.long)
    self.k = k
    self.query_document_token_mapping = query_document_token_mapping
    self.num_doc_tokens = num_doc_tokens

  def __len__(self):
    return len(self.rankings)

  def __getitem__(self, idx):
    query, ranking = self.rankings[idx]
    relevant = set(ranking[:self.k])
    return {'query': torch.tensor(query, dtype=torch.long),
            'documents': self.documents[ranking][:, :self.num_doc_tokens],
            'doc_ids': torch.tensor(ranking, dtype=torch.long),
            'ranking': ranking[:self.k],
            'relevant': relevant}

offset_to_tuple = {}

def _get_nth_pair(rankings, idx):
  offset = idx
  for query, doc_ids in rankings:
    num_doc_ids = len(doc_ids)
    num_pairs_for_query = num_doc_ids ** 2 - num_doc_ids
    if offset < num_pairs_for_query:
      if offset in offset_to_tuple:
        doc_1_idx, doc_2_idx = offset_to_tuple[offset]
      else:
        doc_1_idx = offset // (num_doc_ids - 1)
        doc_2_idx = offset % (num_doc_ids - 1)
        if doc_2_idx >= doc_1_idx:
          doc_2_idx += 1
        offset = (doc_1_idx, doc_2_idx)
      return {'query': query,
              'doc_id_1': doc_ids[doc_1_idx],
              'doc_id_2': doc_ids[doc_2_idx],
              'order_int': 1 if doc_1_idx < doc_2_idx else -1}
    offset -= num_pairs_for_query
  raise IndexError(f'index {idx} out of range for rankings list')

def _get_num_pairs(rankings):
  return reduce(lambda acc, ranking: acc + len(ranking[1]) ** 2 - len(ranking[1]),
                rankings,
                0)

class QueryPairwiseDataset(QueryDataset):
  def __init__(self, documents, data, rel_method=score):
    super().__init__(documents, data)
    self.lowest_rank_doc_to_consider = 100
    self.rankings = _.map_(self.rankings,
                           lambda ranking: [ranking[0], ranking[1][:self.lowest_rank_doc_to_consider]])

  def __len__(self):
    return _get_num_pairs(self.rankings)

  def __getitem__(self, idxs):
    elem = _get_nth_pair(self.rankings, idx)
    return ((elem['query'],
             self.documents[elem['doc_id_1']][:self.num_doc_tokens],
             self.documents[elem['doc_id_2']][:self.num_doc_tokens]),
            elem['order_int'])

def _get_tfidf_transformer_and_matrix(documents):
  transformer = TfidfTransformer()
  counts = sp.lil_matrix((len(documents), documents.max().item() + 1))
  for doc_num, doc in enumerate(documents):
    doc_counts = np.bincount(doc)
    nonzero = doc_counts.nonzero()
    counts[doc_num, nonzero] = doc_counts[nonzero]
  return transformer, transformer.fit_transform(counts)

def score_documents_tfidf(query_document_token_mapping, tfidf_docs, query):
  mapped_query = [query_document_token_mapping[token] for token in query]
  subset = tfidf_docs[:, mapped_query] / (tfidf_docs.sum(1) + 0.0001)
  return torch.tensor(subset.sum(1).T.tolist()).squeeze()

def score_documents_embed(doc_word_embeds, query_word_embeds, documents, queries, device):
  query_embeds = query_word_embeds(queries)
  query_vecs = query_embeds.sum(1).to(device)
  query_vecs = query_vecs / (torch.norm(query_vecs, 2, 1).unsqueeze(1) + 0.0001)
  doc_embeds = doc_word_embeds(documents)
  doc_vecs = doc_embeds.sum(1).to(device)
  doc_vecs = doc_vecs / (torch.norm(doc_vecs, 2, 1).unsqueeze(1) + 0.0001)
  scores = torch.zeros(len(query_vecs), len(doc_vecs), device=device)
  sections = torch.cat([torch.arange(start=0, end=len(doc_vecs), step=1000, dtype=torch.long),
                        torch.tensor([len(doc_vecs)], dtype=torch.long)])
  for chunk_start, chunk_end in zip(sections, sections[1:]):
    logits = (doc_vecs[chunk_start:chunk_end] * query_vecs.unsqueeze(1)).sum(2)
    scores[:, chunk_start:chunk_end] = logits
  return scores

def get_top_k(scores, k=1000):
  sorted_scores, idxs = torch.sort(scores, descending=True)
  return idxs[:k]

def build_query_dataloader(documents, data, rel_method=score) -> DataLoader:
  dataset = QueryDataset(documents, data, rel_method=rel_method)
  return DataLoader(dataset,
                    batch_sampler=BatchSampler(RandomSampler(dataset), 100, False),
                    collate_fn=collate_query_samples)

def build_query_pairwise_dataloader(documents, data, rel_method=score) -> DataLoader:
  dataset = QueryPairwiseDataset(documents, data, rel_method=rel_method)
  return DataLoader(dataset,
                    batch_sampler=BatchSampler(RandomSampler(dataset), 100, False),
                    collate_fn=collate_query_pairwise_samples)
