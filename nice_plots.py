import torch
import pickle
import pydash as _
import matplotlib.pyplot as plt
from lm_ltr.embedding_loaders import get_glove_lookup
import numpy as np

def count_me(docs):
  tf = {0: 0, 1: 0}
  df = {0: 1, 1: 1}
  for doc in docs:
    in_doc = set()
    for idx in doc:
      in_doc.add(idx)
      if idx in tf:
        tf[idx] += 1
      else:
        tf[idx] = 1
    for idx in in_doc:
      if idx in df:
        df[idx] += 1
      else:
        df[idx] = 1
  idf = _.map_values(df, lambda cnt: 1/cnt)
  return tf, df, idf

def doc():
  w = pickle.load(open('weights_from_doc.pkl', 'rb')).squeeze()
  topk_vals, topk_idxs = torch.topk(w, 30)
  bottomk_vals, bottomk_idxs = torch.topk(-w, 30)
  docs, lookup = pickle.load(open('parsed_docs_100_tokens.pkl', 'rb'))
  tf, df, idf = count_me(docs)
  inv_lookup = _.invert(lookup)
  print('Top30: ', [inv_lookup[idx] for idx in topk_idxs.tolist()])
  print('Bottom30: ', [inv_lookup[idx] for idx in bottomk_idxs.tolist()])
  glove = get_glove_lookup()
  glove_by_idx = _.map_keys(glove, lambda vec, token: lookup[token] if token in lookup else lookup['<unk>'])
  norms_by_idx = _.map_values(glove_by_idx, torch.norm)
  idxs_in_order = list(norms_by_idx.keys())
  idfs_in_order = torch.tensor([idf[idx] for idx in idxs_in_order])
  dfs_in_order = torch.tensor([df[idx] for idx in idxs_in_order])
  tfs_in_order = torch.tensor([tf[idx] for idx in idxs_in_order])
  norms_in_order = torch.tensor([norms_by_idx[idx] for idx in idxs_in_order])
  w_subset = w[torch.tensor(idxs_in_order)]
  print(np.corrcoef(w_subset, tfs_in_order)[0, 1])
  print(np.corrcoef(w_subset, dfs_in_order)[0, 1])
  print(np.corrcoef(w_subset, idfs_in_order)[0, 1])
  print(np.corrcoef(w_subset, norms_in_order)[0, 1])
  print(np.corrcoef(w_subset, np.log(tfs_in_order + 1))[0, 1])
  print(np.corrcoef(w_subset, np.log(dfs_in_order))[0, 1])
  print(np.corrcoef(w_subset, np.log(idfs_in_order))[0, 1])
  print(np.corrcoef(w_subset, np.log(norms_in_order + 1))[0, 1])
  plt.scatter(w_subset.numpy(), dfs_in_order.numpy())
  plt.show()
  # plt.hist2d(abs(w_subset.numpy()), norms_in_order.numpy(), bins=100)
  # plt.scatter(abs(w_subset.numpy()), norms_in_order.numpy())
  # plt.scatter(w_subset.numpy(), idfs_in_order.numpy())
  # plt.show()

def query():
  w = pickle.load(open('weights_from_query.pkl', 'rb')).squeeze()
  topk_vals, topk_idxs = torch.topk(w, 30)
  bottomk_vals, bottomk_idxs = torch.topk(-w, 30)
  docs, lookup = pickle.load(open('parsed_robust_queries.pkl', 'rb'))
  tf, df, idf = count_me(docs)
  inv_lookup = _.invert(lookup)
  print('Top30: ', [inv_lookup[idx] for idx in topk_idxs.tolist()])
  print('Bottom30: ', [inv_lookup[idx] for idx in bottomk_idxs.tolist()])
  glove = get_glove_lookup()
  glove_by_idx = _.map_keys(glove, lambda vec, token: lookup[token] if token in lookup else lookup['<unk>'])
  norms_by_idx = _.map_values(glove_by_idx, torch.norm)
  idxs_in_order = list(norms_by_idx.keys())
  idfs_in_order = torch.tensor([idf[idx] for idx in idxs_in_order])
  dfs_in_order = torch.tensor([df[idx] for idx in idxs_in_order])
  tfs_in_order = torch.tensor([tf[idx] for idx in idxs_in_order])
  norms_in_order = torch.tensor([norms_by_idx[idx] for idx in idxs_in_order])
  w_subset = w[torch.tensor(idxs_in_order)]
  print(np.corrcoef(w_subset, tfs_in_order)[0, 1])
  print(np.corrcoef(w_subset, dfs_in_order)[0, 1])
  print(np.corrcoef(w_subset, idfs_in_order)[0, 1])
  print(np.corrcoef(w_subset, norms_in_order)[0, 1])
  print(np.corrcoef(w_subset, np.log(tfs_in_order + 1))[0, 1])
  print(np.corrcoef(w_subset, np.log(dfs_in_order))[0, 1])
  print(np.corrcoef(w_subset, np.log(idfs_in_order))[0, 1])
  print(np.corrcoef(w_subset, np.log(norms_in_order + 1))[0, 1])
  # plt.scatter(w_subset.numpy(), dfs_in_order.numpy())
  # plt.show()
  # plt.hist2d(abs(w_subset.numpy()), norms_in_order.numpy(), bins=100)
  # plt.scatter(abs(w_subset.numpy()), norms_in_order.numpy())
  # plt.scatter(w_subset.numpy(), idfs_in_order.numpy())
  # plt.show()
