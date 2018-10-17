from typing import List, Dict, Tuple
from subprocess import run
import pickle

import pydash as _

from .fetchers import get_wiki_documents_lookup, get_wiki_mention_queries_lookup, get_aol_queries_lookup, load_indri_results, read_cache, get_robust_queries_lookup
from .trec_doc_parse import load_robust04_documents_lookup
from .preprocessing import preprocess_texts
import lm_ltr.create_trectext as create_trectext
from .path_helpers import get_indri_documents_path, get_indri_index_path, get_build_index_path, get_index_params_path

class DataOrganizer:
  document_sources:List[str]
  query_sources:List[str]
  query_results_sources:List[str]
  query_token_lookup:Dict[str,int]    =(lambda:{})()
  document_token_lookup:Dict[str,int] =(lambda:{})()
  document_texts:List[str]            =(lambda:[])()
  query_texts:List[str]               =(lambda:[])()
  documents:List[List[int]]           =(lambda:[])()
  queries:List[List[int]]             =(lambda:[])()
  data:List[Dict[str, List[int]]]     =(lambda:[])()
  robust_paths:List[str]              =(lambda:['./fbis', './la', './ft'])()
  document_id_lookup:Dict[str,int]    =(lambda:{})()
  query_id_lookup:Dict[str,int]       =(lambda:{})()

  def __post_init__(self):
    if len(self.document_sources) > 1 or len(self.query_sources) > 1:
      raise NotImplementedError('Only single document and query sources supported')

  @property
  def _cache_path(self):
    return f'./data_organizer_cache_{"_".join(sorted(self.query_sources))}_{"_".join(sorted(self.document_sources))}.pkl'

  def _update_id_lookup(self, id_lookup, titles_in_order) -> None:
    from_id = len(id_lookup)
    to_id = from_id + len(titles_in_order)
    new_doc_ids = range(from_id, to_id)
    id_lookup.update(dict(zip(titles_in_order,
                              new_doc_ids)))

  def _load_documents(self) -> None:
    def _load_documents_from(source) -> List[List[str]]:
      if source == 'wiki':
        documents_lookup = get_wiki_documents_lookup()
      elif source == 'robust04':
        documents_lookup = load_robust04_documents_lookup(self.robust_paths)
      elif source == 'clueweb':
        raise NotImplementedError('no work done for clueweb yet')
      else:
        raise ValueError(source + ' is not a valid document source')
      document_titles_in_order = list(documents_lookup.keys())
      self._update_id_lookup(self.document_id_lookup, document_titles_in_order)
      return [documents_lookup[doc_title] for doc_title in document_titles_in_order]
    for source in self.document_sources:
      self.document_texts += _load_documents_from(source)

  def _load_queries(self) -> None:
    def _load_queries_from(source):
      if source == 'wiki_mentions':
        # queries_lookup = get_wiki_mention_queries_lookup()
        raise NotImplementedError
      elif source == 'aol':
        queries_lookup = get_aol_queries_lookup()
      elif source == 'robust04':
        queries_lookup = get_robust_queries_lookup()
      else:
        raise ValueError(source + ' is not a valid query source')
      query_names_in_order = list(queries_lookup.keys())
      self._update_id_lookup(self.query_id_lookup, query_names_in_order)
      return [queries_lookup[query_name] for query_name in query_names_in_order]
    for source in self.query_sources:
      self.query_texts += _load_queries_from(source)

  def _create_one_hot_documents(self) -> None:
    doc_ids = sorted(list(self.document_id_lookup.values()))
    old_doc_id_lookup = _.invert(self.document_id_lookup)
    contents = [self.document_texts[old_doc_id_lookup[doc_id]] for doc_id in doc_ids]
    self.documents, self.document_token_lookup = preprocess_texts(contents)

  def _create_one_hot_queries(self) -> None:
    query_ids = sorted(list(self.query_id_lookup.values()))
    query_name_lookup = _.invert(self.query_id_lookup)
    contents = [self.query_texts[query_name_lookup[query_id]] for query_id in query_ids]
    self.queries, self.query_token_lookup = preprocess_texts(contents)

  def _query_results_to_data(self, query_results):
    return [{'query': self.queries[row['query_id']],
             'doc_id': row['doc_id'],
             'score': row['score']} for row in query_results]

  def _create_indri_documents(self):
    for source in self.document_sources:
      path = get_indri_documents_path(self.document_sources)
      if source == 'wiki':
        create_trectext.from_wikipedia(self.document_texts,
                                       self.document_id_lookup,
                                       path)

  def _build_indri_index(self):
    try:
      run(['rm', '-rf', get_indri_index_path()])
      run(['mkdir', get_indri_index_path()])
    except OSError:
      pass
    run([get_build_index_path(),
         get_index_params_path()]).check_returncode()

  def _create_indri_queries(self):
    pass

  @property
  def indexed_document_sources(self):
    try:
      with open('./indexed_doument_sources.pkl', 'rb') as fh:
        return pickle.load(fh)
    except FileNotFoundError:
      return []

  def _update_indexed_document_sources(self):
    with open('./indexed_doument_sources.pkl', 'wb+') as fh:
      pickle.dump(self.document_sources, fh)

  def _load_query_results(self) -> None:
    def _load_query_results_from(results_source):
      if results_source == 'indri':
        self._create_indri_queries()
        self._create_indri_documents()
        if not _.is_empty(set(self.document_sources) - set(self.indexed_document_sources)):
          self._build_indri_index()
          self._update_indexed_document_sources()
        query_results = load_indri_results(self.query_sources, self.document_sources)
      elif results_source == 'mention_occurrence':
        raise NotImplementedError('no work done for wiki mention occurrence yet')
      else:
        raise ValueError(results_source + ' is not a valid query results source')
      return query_results
    for results_source in self.query_results_sources:
      self.data = _load_query_results_from(results_source)

  def _load(self) -> Tuple[List[Dict[str, List[int]]], List[List[int]], List[List[int]]]:
    self._load_documents()
    self._load_queries()
    self._create_one_hot_documents()
    self._create_one_hot_queries()
    self._load_query_results()
    return self.data, self.documents, self.queries

  def load_all(self) -> Tuple[List[Dict[str, List[int]]], List[List[int]], List[List[int]]]:
    return read_cache(self._cache_path, self._load)
