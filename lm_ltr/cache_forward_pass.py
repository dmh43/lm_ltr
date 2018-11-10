import torch
import json

from lm_ltr.fetchers import read_from_file
from lm_ltr.pretrained import get_doc_encoder_and_embeddings
from lm_ltr.utils import dont_update
from lm_ltr.preprocessing import pad

def main():
  documents, document_token_lookup = read_from_file('./parsed_docs_500_tokens_limit_uniq_toks.json')
  doc_encoder, document_token_embeds = get_doc_encoder_and_embeddings(document_token_lookup)
  doc_encoder = doc_encoder.to(torch.device("cuda"))
  dont_update(doc_encoder)
  batches = []
  for from_idx, to_idx in zip(range(int(len(documents) / 1000)),
                              range(1000, int(len(documents) / 1000) + 1000)):
    doc_batch = [torch.tensor(doc[:500]) for doc in documents[from_idx:to_idx]]
    batches.append(doc_encoder(pad(doc_batch)).tolist())
  with open('./forward_out.json', 'w+') as fh:
    json.dump(batches, fh)


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
