import random

import pydash as _

def append_at(obj, path, val):
  if _.has(obj, path):
    _.update(obj, path, lambda coll: coll + [val])
  else:
    _.set_(obj, path, [val])

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
