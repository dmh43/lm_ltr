import torch

from lm_ltr.pointwise_ranker import PointwiseRanker

class MyScorer:
  def __init__(self):
    self.training = False

  def __call__(self, encoded_query, encoded_documents, lens):
    return torch.sum(encoded_query * encoded_documents.float(), dim=1)

def test_pointwise_ranker():
  batch_size = 10
  num_docs = 15
  embed_size = 100
  scorer = MyScorer()
  encoded_query = torch.randn(batch_size, embed_size)
  encoded_documents = torch.cat([encoded_query,
                                 torch.randn(num_docs - batch_size, embed_size)], dim=0)
  ranker = PointwiseRanker(torch.device('cpu'), scorer)
  ranking = ranker(encoded_query, encoded_documents)
  assert torch.equal(ranking[:, 0], torch.arange(batch_size, dtype=torch.long))
