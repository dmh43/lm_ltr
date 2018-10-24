import torch
from lm_ltr.metrics import RankingMetricRecorder, metrics_at_k

def test_ranking_metrics_at_k():
  train_ranking_dl = None
  test_ranking_dl = None
  scorer = lambda query, documents: - (query[:, 0] - torch.nn.utils.rnn.pad_packed_sequence(documents[0],
                                                                                            padding_value=0,
                                                                                            batch_first=True)[0].nonzero()[:, 0]) ** 2
  device = torch.device('cpu')
  metric = RankingMetricRecorder(device, scorer, train_ranking_dl, test_ranking_dl, None)
  num_documents = 10
  num_queries = 10
  documents = torch.eye(num_documents)
  dataset = [{'query': torch.tensor([i]),
              'documents': documents,
              'relevant': list(range(num_documents))[i - 1 : i + 2],
              'doc_ids': torch.arange(len(documents), dtype=torch.long)} for i in range(num_queries)]
  assert metric.metrics_at_k(dataset, k=1) == {'precision': 1.0, 'recall': 9.0 / 26, 'ndcg': 1.0, 'map': (8/3 + 1/2) / 9}
  assert metric.metrics_at_k(dataset, k=3) == {'precision': 26.0 / 27, 'recall': 1.0, 'ndcg': 1.0, 'map': 1.0}
  assert metric.metrics_at_k(dataset, k=6) == {'precision': 26.0 / 54, 'recall': 1.0, 'ndcg': 1.0, 'map': 1.0}

def test_metrics_at_k():
  rankings_to_judge = [[1, 2, 3, 4], [1, 2, 3, 9], [3, 4, 7, 10]]
  relevant_doc_ids = [[1, 2, 3, 4], [1, 2, 3, 9], [3, 4, 7, 10]]
  expected = {'precision': 1.0, 'recall': 1.0, 'ndcg': 1.0, 'map': 1.0}
  result = metrics_at_k(rankings_to_judge, relevant_doc_ids, k=4)
  assert expected == result
