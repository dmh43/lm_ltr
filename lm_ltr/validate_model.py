import torch
import numpy as np

from fastai.core import VV, to_np, datafy
from fastai.model import batch_sz

def evaluate_model(model, crit, xs, y):
  preds = model(*xs)
  if isinstance(preds,tuple): preds=preds[0]
  return preds, crit(preds, y)

def validate_model(model, crit, dl, metrics):
  batch_cnts,loss,res = [],[],[]
  with torch.no_grad():
    for (*x,y) in iter(dl):
      y = VV(y)
      preds, l = evaluate_model(model, crit, VV(x), y)
      batch_cnts.append(batch_sz(x))
      loss.append(to_np(l))
      res.append([f(datafy(preds), datafy(y)) for f in metrics])
  return [np.average(loss, 0, weights=batch_cnts)] + list(np.average(np.stack(res), 0, weights=batch_cnts))
