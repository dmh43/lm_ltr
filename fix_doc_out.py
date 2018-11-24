import json

def main():
  with open('./forward_out.json') as fh:
    outs = json.load(fh)
  with open('./forward_out_flat.json', 'w+') as fh:
    json.dump(sum(outs, []), fh)

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
