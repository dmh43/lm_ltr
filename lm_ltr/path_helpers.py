root = '/home/dany/lm_ltr'
bin_root = '/home/dany/bin/bin'

def get_indri_documents_path(document_sources):
  return f'{root}/indri/docs_{"_".join(sorted(document_sources))}.xml'

def get_indri_index_path():
  return f'{root}/indri/index'

def get_build_index_path():
  return f'{bin_root}/IndriBuildIndex'

def get_run_query_path():
  return f'{bin_root}/IndriRunQuery'

def get_index_params_path():
  return f'{root}/index_params.xml'

def get_indri_result_path(query_source, document_sources):
  return f'./indri/query_result_{query_source}_{"_".join(sorted(document_sources))}'

def get_indri_queries_path(query_source):
  return f'./indri/query_params_{query_source}.xml'
