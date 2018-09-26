import torch
from lm_ltr.metrics import RankingMetricRecorder

def test_precision_at_k():
  train_ranking_dl = None
  test_ranking_dl = None
  scorer = lambda query, documents: - (query[:, 0] - documents.nonzero()[:, 0]) ** 2
  metric = RankingMetricRecorder(scorer, train_ranking_dl, test_ranking_dl)
  num_documents = 10
  num_queries = 10
  documents = torch.eye(num_documents)
  dataset = [{'query': torch.tensor([i]),
              'documents': documents,
              'relevant': list(range(num_documents))[i - 1 : i + 2]} for i in range(num_queries)]
  assert metric.precision_at_k(dataset, k=1) == 0.9
  assert metric.precision_at_k(dataset, k=3) == 26.0 / 30
  assert metric.precision_at_k(dataset, k=6) == 26.0 / 60
