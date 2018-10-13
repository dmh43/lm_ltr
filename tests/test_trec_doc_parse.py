import pydash as _

import lm_ltr.trec_doc_parse as parse

def test__parse_qrels():
  result = parse._parse_qrels('./tests/fixtures/qrels')
  assert result == {'301': list(reversed(['FBIS3-10491',
                                          'FBIS3-10397',
                                          'FBIS3-10243',
                                          'FBIS3-10082']))}

def test__parse_test_set():
  result = parse._parse_test_set('./tests/fixtures/test_set')
  assert result == {'301': 'International Organized Crime',
                    '302': 'Poliomyelitis and Post-Polio'}

def test_parse_robust():
  qrels_path = './tests/fixtures/qrels'
  test_set_path = './tests/fixtures/test_set'
  doc_paths = ['./tests/fixtures/fbis_sample']
  num_docs_in_fbis_sample = 5
  query_token_lookup = {'international': 2, 'organized': 3}
  document_token_lookup = {}
  data, docs = parse.parse_robust(query_token_lookup,
                                  document_token_lookup,
                                  qrels_path,
                                  test_set_path,
                                  doc_paths,
                                  doc_first_index=1)
  print(docs)
  assert all([row['doc_id'] >= 1 for row in data])
  assert len(docs) == num_docs_in_fbis_sample
  assert data == [{'query': [2, 3, 0], 'doc_id': 4},
                  {'query': [2, 3, 0], 'doc_id': 3},
                  {'query': [2, 3, 0], 'doc_id': 2},
                  {'query': [2, 3, 0], 'doc_id': 1}]
