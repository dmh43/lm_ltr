import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler

import scipy.sparse as sp
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer

from preprocessing import collate_query_samples, to_query_rankings_pairs, pad_to_max_len, all_ones, score, inv_log_rank, inv_rank, exp_score

class QueryDataset(Dataset):
  def __init__(self, documents, data, rel_method=score):
    self.documents = documents
    self.data = data
    self.rel_method = rel_method

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return ((self.data[idx]['query'], self.documents[self.data[idx]['document_id']]),
            self.rel_method(self.data[idx]))

def _get_tfidf_transformer_and_matrix(documents):
  transformer = TfidfTransformer()
  counts = sp.lil_matrix((len(documents), documents.max().item() + 1))
  for doc_num, doc in enumerate(documents):
    doc_counts = np.bincount(doc)
    nonzero = doc_counts.nonzero()
    counts[doc_num, nonzero] = doc_counts[nonzero]
  return transformer, transformer.fit_transform(counts)

def _get_top_scoring(tfidf_docs, query, k=1000):
  subset = tfidf_docs[:, query]
  scores = torch.tensor(subset.sum(1).T.tolist()).squeeze()
  sorted_scores, idxs = torch.sort(scores, descending=True)
  return idxs[:k]

def score_documents_tfidf(query_document_token_mapping, tfidf_docs, query):
  mapped_query = [query_document_token_mapping[token] for token in query]
  subset = tfidf_docs[:, mapped_query]
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

class RankingDataset(Dataset):
  def __init__(self, documents, data, query_document_token_mapping, k=10):
    self.documents = torch.tensor(pad_to_max_len(documents), dtype=torch.long)
    self.k = k
    self.rankings = to_query_rankings_pairs(data, k=self.k)
    self.tfidf_transformer, self.tfidf_docs = _get_tfidf_transformer_and_matrix(self.documents)
    self.query_document_token_mapping = query_document_token_mapping

  def __len__(self):
    return len(self.rankings)

  def __getitem__(self, idx):
    query, ranking = self.rankings[idx]
    remapped_query = torch.tensor([self.query_document_token_mapping[token] for token in query], dtype=torch.long)
    ranking = torch.tensor(ranking, dtype=torch.long)
    return {'query': torch.tensor(query, dtype=torch.long),
            'documents': self.documents[_get_top_scoring(self.tfidf_docs, remapped_query)],
            'ranking': ranking,
            'relevant': ranking}

def build_query_dataloader(documents, data, rel_method=score) -> DataLoader:
  dataset = QueryDataset(documents, data, rel_method=rel_method)
  return DataLoader(dataset,
                    batch_sampler=BatchSampler(RandomSampler(dataset), 100, False),
                    collate_fn=collate_query_samples)
