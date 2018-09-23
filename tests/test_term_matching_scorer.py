import torch

from lm_ltr.term_matching_scorer import TermMatchingScorer

def test_term_matching_scorer():
  query_document_token_mapping = {i: i for i in range(10)}
  scorer = TermMatchingScorer(query_document_token_mapping)
  scorer.weights.data = torch.ones_like(scorer.weights)
  counts = torch.tensor([[3, 3, 3, 3],
                         [1, 1, 1, 1]])
  terms = torch.tensor([[0, 1, 3, 4],
                        [0, 1, 3, 4]])
  score = scorer(counts, terms)
  assert score[0] > score[1]
