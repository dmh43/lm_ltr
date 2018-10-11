import torch

from lm_ltr.pointwise_ranker import PointwiseRanker

def test_pointwise_ranker():
  batch_size = 10
  num_docs = 15
  embed_size = 100
  scorer = lambda encoded_query, encoded_documents: torch.sum(encoded_query * torch.nn.utils.rnn.pad_packed_sequence(encoded_documents[0], batch_first=True)[0][encoded_documents[1]].float(), dim=1)
  encoded_query = torch.randn(batch_size, embed_size)
  encoded_documents = torch.cat([encoded_query,
                                 torch.randn(num_docs - batch_size, embed_size)], dim=0)
  ranker = PointwiseRanker(torch.device('cpu'), scorer)
  ranking = ranker(encoded_query, encoded_documents)
  assert torch.equal(ranking[:, 0], torch.arange(batch_size, dtype=torch.long))
