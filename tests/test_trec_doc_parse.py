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
