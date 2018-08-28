import torch

def get_glove_lookup(path='./glove/glove.6B.100d.txt', embedding_dim=100):
  lookup = {'<pad>': torch.zeros(size=(embedding_dim,), dtype=torch.float32),
            '<unk>': torch.randn(size=(embedding_dim,), dtype=torch.float32)}
  with open(path) as f:
    while True:
      line = f.readline()
      if line and len(line) > 0:
        split_line = line.strip().split(' ')
        lookup[split_line[0]] = torch.tensor([float(val) for val in split_line[1:]],
                                             dtype=torch.float32)
      else:
        break
  return lookup
