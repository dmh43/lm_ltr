import torch
import json

from lm_ltr.fetchers import read_from_file
from lm_ltr.pretrained import get_doc_encoder_and_embeddings
from lm_ltr.utils import dont_update
from lm_ltr.preprocessing import pad

def main():
  documents, document_token_lookup = read_from_file('./parsed_docs_500_tokens_limit_uniq_toks.json')
  doc_encoder, document_token_embeds = get_doc_encoder_and_embeddings(document_token_lookup)
  device = torch.device("cuda")
  doc_encoder = doc_encoder.to(device)
  dont_update(doc_encoder)
  results = []
  for from_idx, to_idx in zip(range(0, int(len(documents) / 200), 200),
                              range(200, int(len(documents) / 200) + 200, 200)):
    doc_batch = [torch.tensor(doc[:500]) for doc in documents[from_idx:to_idx]]
    padded_batch, lens = pad(doc_batch)
    sorted_lens, sort_order = torch.sort(lens, descending=True)
    batch_range, unsort_order = torch.sort(sort_order)
    sorted_batch = padded_batch[sort_order]
    seq_dim_first = torch.transpose(sorted_batch, 0, 1).to(device)
    results.extend(doc_encoder(seq_dim_first)[unsort_order].tolist())
  with open('./forward_out.json', 'w+') as fh:
    json.dump(results, fh)


if __name__ == "__main__":
  import ipdb
  import traceback
  import sys

  try:
    main()
  except: # pylint: disable=bare-except
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
  ipdb.post_mortem(tb)
