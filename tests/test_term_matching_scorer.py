import torch

from lm_ltr.term_matching_scorer import TermMatchingScorer

def test_term_matching_scorer_score():
  query_document_token_mapping = {i: i for i in range(10)}
  scorer = TermMatchingScorer(query_document_token_mapping)
  scorer.weights.data = torch.ones_like(scorer.weights)
  counts = torch.tensor([[3, 3, 3, 3],
                         [1, 1, 1, 1]])
  terms = torch.tensor([[0, 1, 3, 4],
                        [0, 1, 3, 4]])
  score = scorer.score(counts, terms)
  assert score[0] > score[1]

def test_term_matching_scorer_get_counts_terms():
  query_document_token_mapping = {i: i for i in range(10)}
  scorer = TermMatchingScorer(query_document_token_mapping)
  query = torch.tensor([[0, 2, 0]], dtype=torch.long)
  document = torch.arange(3, dtype=torch.long).repeat(10).unsqueeze(0)
  expected_counts = torch.tensor([[10, 10, 10]], dtype=torch.long)
  expected_terms = torch.tensor([[0, 2, 0]], dtype=torch.long)
  counts, terms = scorer.get_counts_terms(query, document)
  assert torch.equal(counts, expected_counts)
  assert torch.equal(terms, expected_terms)
