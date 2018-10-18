import pydash as _
import lm_ltr.fetchers as f

def test_parse_xml_docs_fbis():
  doc_path = './tests/fixtures/fbis_sample'
  doc_lookup = f.parse_xml_docs(doc_path)
  assert len(doc_lookup) == 5
  assert _.is_empty(set(doc_lookup.keys()) - set(['FBIS3-10491',
                                                  'FBIS3-10397',
                                                  'FBIS3-10243',
                                                  'FBIS3-10082',
                                                  'FBIS3-5']))
  assert all([len(text) > 1000 for text in doc_lookup.values()])

def test_parse_xml_docs_la():
  doc_path = './tests/fixtures/la_sample'
  doc_lookup = f.parse_xml_docs(doc_path)
  assert len(doc_lookup) == 1
  assert _.is_empty(set(doc_lookup.keys()) - set(['LA010189-0001']))
  assert all([len(text) > 100 for text in doc_lookup.values()])

def test_parse_xml_docs_trouble_fbis():
  doc_path = './tests/fixtures/trouble_fbis'
  doc_lookup = f.parse_xml_docs(doc_path)
  assert len(doc_lookup) == 2
  assert _.is_empty(set(doc_lookup.keys()) - set(['FBIS3-10491', 'FBIS3-10081']))
  assert all([len(text) > 100 for text in doc_lookup.values()])
