import torch
from lm_ltr.metrics import RankingMetricRecorder

def test_metrics_at_k():
  train_ranking_dl = None
  test_ranking_dl = None
  scorer = lambda query, documents: - (query[:, 0] - torch.nn.utils.rnn.pad_packed_sequence(documents[0],
                                                                                            padding_value=0,
                                                                                            batch_first=True)[0].nonzero()[:, 0]) ** 2
  device = torch.device('cpu')
  metric = RankingMetricRecorder(device, scorer, train_ranking_dl, test_ranking_dl)
  num_documents = 10
  num_queries = 10
  documents = torch.eye(num_documents)
  dataset = [{'query': torch.tensor([i]),
              'documents': documents,
              'relevant': list(range(num_documents))[i - 1 : i + 2],
              'doc_ids': torch.arange(len(documents), dtype=torch.long)} for i in range(num_queries)]
  assert metric.metrics_at_k(dataset, k=1) == (1.0, 9.0 / 26, 1.0)
  assert metric.metrics_at_k(dataset, k=3) == (26.0 / 27, 1.0, 1.0)
  assert metric.metrics_at_k(dataset, k=6) == (26.0 / 54, 1.0, 1.0)
