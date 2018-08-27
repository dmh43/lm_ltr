from typing import List

import torch
from fastai.text import Tokenizer

pad_token_idx = 1
unk_token_idx = 0

def tokens_to_indexes(tokens: List[List[str]], lookup=None):
  is_test = lookup is not None
  if lookup is None:
    lookup: dict = {'<unk>': unk_token_idx, '<pad>': pad_token_idx}
  result = []
  for tokens_chunk in tokens:
    chunk_result = []
    for token in tokens_chunk:
      if is_test:
        chunk_result.append(lookup.get(token) or unk_token_idx)
      else:
        lookup[token] = lookup.get(token) or len(lookup)
        chunk_result.append(lookup[token])
    result.append(chunk_result)
  return result, lookup

def preprocess_texts(texts: List[str], token_lookup=None):
  tokenizer = Tokenizer()
  tokenized = [tokenizer.proc_text(q) for q in texts]
  idx_texts, token_lookup = tokens_to_indexes(tokenized, token_lookup)
  return idx_texts, token_lookup

def pad_to_max_len(elems: List[List[int]]):
  max_len = max(map(len, elems))
  return [elem + [pad_token_idx] * (max_len - len(elem)) if len(elem) < max_len else elem for elem in elems]

def collate(samples):
  x, y = list(zip(*samples))
  x = list(zip(*x))
  return torch.tensor(pad_to_max_len(x[0])), x[1], torch.tensor(y)
