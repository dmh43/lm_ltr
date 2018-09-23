import torch
import lm_ltr.preprocessing as p

def test_to_ranking_per_query():
  data = [{'query': [1, 2],
           'document_id': i*10,
           'rank': i} for i in range(10)]
  assert p.to_query_rankings_pairs(data) == [[[1, 2], list(range(0, 100, 10))]]

def test_get_term_matching():
  query_document_token_mapping = {i: i for i in range(10)}
  query = torch.tensor([0, 2, 0])
  document = torch.arange(3, dtype=torch.long).repeat(10)
  expected_counts = torch.tensor([10, 10, 10])
  expected_terms = torch.tensor([0, 2, 0])
  counts, terms = p.get_term_matching(query_document_token_mapping, query, document)
  assert torch.equal(counts, expected_counts)
  assert torch.equal(terms, expected_terms)
