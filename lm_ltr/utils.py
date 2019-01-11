import torch
import torch.nn as nn

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import random

import pydash as _

def append_at(obj, key, val):
  if key in obj:
    obj[key].append(val)
  else:
    obj[key] = [val]

def with_negative_samples(samples, num_negative_samples, num_documents):
  def _get_neg_samples(sample, num_negative_samples, num_documents):
    return [_.assign({},
                     sample,
                     {'title_id': random.randint(0, num_documents - 1),
                      'rel': 0.0}) for i in range(num_negative_samples)]
  result = samples
  for sample in samples:
    result = result + _get_neg_samples(sample, num_negative_samples, num_documents)
  return result

def dont_update(module):
  for p in module.parameters():
    p.requires_grad = False

def name(path, notes):
  if len(notes) == 0: return path
  path_segs = path.split('.json')
  return '_'.join([path_segs[0]] + notes) + '.json'

def do_update(module):
  for p in module.parameters():
    p.requires_grad = True

def plots(model, im_path):
  norms = model['pointwise_scorer.document_encoder.document_token_embeds.weight'].norm(dim=1)
  no = norms[norms > 2].cpu().numpy()
  w = model['pointwise_scorer.document_encoder.weights.weight'].detach().cpu().squeeze()[norms > 2].numpy()
  plt.hist2d(no, w, bins=100)
  plt.savefig(im_path + '/doc.png')
  plt.figure()
  norms = model['pointwise_scorer.query_encoder.query_token_embeds.weight'].norm(dim=1)
  no = norms[norms > 2].cpu().numpy()
  w = model['pointwise_scorer.query_encoder.weights.weight'].detach().cpu().squeeze()[norms > 2].numpy()
  plt.hist2d(no, w, bins=100)
  plt.savefig(im_path + '/q.png')
  plt.figure()

class Identity(nn.Module):
  def forward(self, x): return x

def at_least_one_dim(tensor):
  if len(tensor.shape) == 0:
    return tensor.unsqueeze(0)
  else:
    return tensor

def to_list(coll):
  if isinstance(coll, torch.Tensor):
    return coll.tolist()
  else:
    return list(coll)
