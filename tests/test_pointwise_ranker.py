import torch

from lm_ltr.pointwise_ranker import PointwiseRanker

def test_pointwise_ranker():
  batch_size = 10
  num_docs = 15
  embed_size = 100
  scorer = lambda query, documents: torch.sum(query * documents, dim=1)
  query = torch.randn(batch_size, embed_size)
  documents = torch.cat([query,
                         torch.randn(num_docs - batch_size, embed_size)], dim=0)
  ranker = PointwiseRanker(scorer, documents)
  ranking = ranker(query)
  assert torch.equal(ranking[:, 0], torch.arange(batch_size, dtype=torch.long))
